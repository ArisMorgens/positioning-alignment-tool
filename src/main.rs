use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use anyhow::Result;
use slint::{ComponentHandle, Model};
use tokio::sync::{broadcast, mpsc};

mod renderer;

slint::include_modules!();

// ── TOC cache (same pattern as swarmkeeper) ──────────────────────────────────

#[derive(Clone)]
struct FileTocCache {
    cache_dir: std::path::PathBuf,
}

impl FileTocCache {
    fn new() -> Self {
        let cache_dir = dirs_next::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("loco-lh-alignment-tool")
            .join("toc_cache");
        std::fs::create_dir_all(&cache_dir).ok();
        FileTocCache { cache_dir }
    }

    fn path_for(&self, key: &[u8]) -> std::path::PathBuf {
        let hex: String = key.iter().map(|b| format!("{:02x}", b)).collect();
        self.cache_dir.join(format!("{}.json", hex))
    }
}

impl crazyflie_lib::TocCache for FileTocCache {
    fn get_toc(&self, key: &[u8]) -> Option<String> {
        std::fs::read_to_string(self.path_for(key)).ok()
    }

    fn store_toc(&self, key: &[u8], toc: &str) {
        std::fs::write(self.path_for(key), toc).ok();
    }
}

// ── Domain types ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum SystemKind {
    Loco,
    Lighthouse,
}

impl SystemKind {
    fn from_label(s: &str) -> Self {
        if s.eq_ignore_ascii_case("lighthouse") {
            SystemKind::Lighthouse
        } else {
            SystemKind::Loco
        }
    }

    fn label(&self) -> &'static str {
        match self {
            SystemKind::Loco => "Loco",
            SystemKind::Lighthouse => "Lighthouse",
        }
    }

    /// (lighthouse.fwdToEstimator, loco.fwdToEstimator) for isolating this system in the estimator.
    fn estimator_sources(&self) -> (u8, u8) {
        match self {
            SystemKind::Lighthouse => (1, 0),
            SystemKind::Loco => (0, 1),
        }
    }
}

#[derive(Clone)]
struct MeasurementConfig {
    base_kind: SystemKind,
    moving_kind: SystemKind,
    num_points: usize,
    num_samples: usize,
    settle_time_s: f64,
}

#[derive(Clone, Default)]
struct PositionStats {
    mean_x: f32,
    mean_y: f32,
    mean_z: f32,
    std_x: f32,
    std_y: f32,
    std_z: f32,
}

impl PositionStats {
    fn from_vecs(xs: Vec<f32>, ys: Vec<f32>, zs: Vec<f32>) -> Self {
        let n = xs.len();
        if n == 0 {
            return Self::default();
        }
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let stddev = |v: &[f32]| {
            if v.len() < 2 {
                return 0.0f32;
            }
            let m = mean(v);
            let var: f32 = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (v.len() - 1) as f32;
            var.sqrt()
        };
        Self {
            mean_x: mean(&xs),
            mean_y: mean(&ys),
            mean_z: mean(&zs),
            std_x: stddev(&xs),
            std_y: stddev(&ys),
            std_z: stddev(&zs),
        }
    }

    /// Combined (Euclidean norm) standard deviation across all three axes —
    /// a single number summarizing the spread of this capture's samples.
    fn total_std(&self) -> f32 {
        (self.std_x.powi(2) + self.std_y.powi(2) + self.std_z.powi(2)).sqrt()
    }
}

// ── Live position state (continuous logging) ─────────────────────────────────

/// Window over which the live standard deviation is computed; samples older
/// than this are dropped from `LiveState::buffer`.
const LIVE_WINDOW: Duration = Duration::from_secs(5);

/// Half-extent (in meters) of the ground grid drawn in the live 3D view.
const VIZ_GRID_RANGE: i32 = 8;

#[derive(Clone, Copy)]
struct LiveSample {
    t: Instant,
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Default)]
struct LiveState {
    buffer: VecDeque<LiveSample>,
}

/// Appends a sample and drops any samples older than `LIVE_WINDOW`.
fn push_live_sample(state: &mut LiveState, sample: LiveSample) {
    state.buffer.push_back(sample);
    while let Some(front) = state.buffer.front() {
        if sample.t.duration_since(front.t) > LIVE_WINDOW {
            state.buffer.pop_front();
        } else {
            break;
        }
    }
}

/// Sample standard deviation of x/y/z over the current buffer (0 if fewer than 2 samples).
fn live_std(state: &LiveState) -> (f32, f32, f32) {
    let n = state.buffer.len();
    if n < 2 {
        return (0.0, 0.0, 0.0);
    }
    let mean_x = state.buffer.iter().map(|s| s.x).sum::<f32>() / n as f32;
    let mean_y = state.buffer.iter().map(|s| s.y).sum::<f32>() / n as f32;
    let mean_z = state.buffer.iter().map(|s| s.z).sum::<f32>() / n as f32;
    let var = |mean: f32, get: fn(&LiveSample) -> f32| {
        state.buffer.iter().map(|s| (get(s) - mean).powi(2)).sum::<f32>() / (n - 1) as f32
    };
    (
        var(mean_x, |s| s.x).sqrt(),
        var(mean_y, |s| s.y).sqrt(),
        var(mean_z, |s| s.z).sqrt(),
    )
}

#[derive(Clone)]
struct PointResult {
    point_index: usize,
    base: PositionStats,
    moving: PositionStats,
}

impl PointResult {
    fn dx(&self) -> f32 { self.base.mean_x - self.moving.mean_x }
    fn dy(&self) -> f32 { self.base.mean_y - self.moving.mean_y }
    fn dz(&self) -> f32 { self.base.mean_z - self.moving.mean_z }
}

// ── Serde output types ────────────────────────────────────────────────────────

#[derive(serde::Serialize)]
struct AlignmentResult {
    base_system: String,
    moving_system: String,
    shift_vector: XyzValue,
    std_deviation: XyzValue,
    point_count: usize,
    points: Vec<PointResultYaml>,
}

#[derive(Clone, Copy, serde::Serialize)]
struct XyzValue {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(serde::Serialize)]
struct PointResultYaml {
    point: usize,
    base: XyzValue,
    moving: XyzValue,
    shift: XyzValue,
}

fn build_alignment_result(
    results: &[PointResult],
    base_kind: SystemKind,
    moving_kind: SystemKind,
) -> AlignmentResult {
    let n = results.len() as f32;
    let mean_dx = results.iter().map(|r| r.dx()).sum::<f32>() / n;
    let mean_dy = results.iter().map(|r| r.dy()).sum::<f32>() / n;
    let mean_dz = results.iter().map(|r| r.dz()).sum::<f32>() / n;

    let stddev_axis = |values: &[f32]| {
        if values.len() < 2 {
            return 0.0f32;
        }
        let m = values.iter().sum::<f32>() / values.len() as f32;
        let var = values.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (values.len() - 1) as f32;
        var.sqrt()
    };
    let dxs: Vec<f32> = results.iter().map(|r| r.dx()).collect();
    let dys: Vec<f32> = results.iter().map(|r| r.dy()).collect();
    let dzs: Vec<f32> = results.iter().map(|r| r.dz()).collect();

    // Shift to apply to the moving system's positions so they line up with the base system.
    let shift_vector = XyzValue { x: mean_dx, y: mean_dy, z: mean_dz };

    AlignmentResult {
        base_system: base_kind.label().to_string(),
        moving_system: moving_kind.label().to_string(),
        shift_vector,
        std_deviation: XyzValue {
            x: stddev_axis(&dxs),
            y: stddev_axis(&dys),
            z: stddev_axis(&dzs),
        },
        point_count: results.len(),
        points: results
            .iter()
            .map(|r| PointResultYaml {
                point: r.point_index + 1,
                base: XyzValue { x: r.base.mean_x, y: r.base.mean_y, z: r.base.mean_z },
                moving: XyzValue { x: r.moving.mean_x, y: r.moving.mean_y, z: r.moving.mean_z },
                shift: XyzValue { x: r.dx(), y: r.dy(), z: r.dz() },
            })
            .collect(),
    }
}

// ── Moving-system position file shifting ──────────────────────────────────────

#[derive(serde::Deserialize, serde::Serialize)]
struct AnchorPos {
    x: f32,
    y: f32,
    z: f32,
}

/// Loco anchor positions YAML: `{<anchor id>: {x, y, z}, ...}`.
fn shift_anchor_positions_yaml(content: &str, shift: XyzValue) -> Result<String> {
    let mut anchors: std::collections::BTreeMap<i64, AnchorPos> = serde_yaml::from_str(content)?;
    for pos in anchors.values_mut() {
        pos.x += shift.x;
        pos.y += shift.y;
        pos.z += shift.z;
    }
    Ok(serde_yaml::to_string(&anchors)?)
}

/// Lighthouse base station geometry YAML: `{geos: {<bs id>: {origin: [x,y,z], rotation: [[..]]}}, ...}`.
/// Only the `origin` positions are shifted; rotation matrices and any `calibs` section are left untouched.
fn shift_lh_geometry_yaml(content: &str, shift: XyzValue) -> Result<String> {
    let mut value: serde_yaml::Value = serde_yaml::from_str(content)?;
    let geos = value
        .get_mut("geos")
        .ok_or_else(|| anyhow::anyhow!("YAML has no 'geos' section — expected a Lighthouse base station geometry file"))?;
    let map = geos
        .as_mapping_mut()
        .ok_or_else(|| anyhow::anyhow!("'geos' is not a mapping"))?;

    let deltas = [shift.x as f64, shift.y as f64, shift.z as f64];
    for (_id, geo) in map.iter_mut() {
        let origin = geo
            .get_mut("origin")
            .and_then(|o| o.as_sequence_mut())
            .ok_or_else(|| anyhow::anyhow!("a base station entry is missing 'origin'"))?;
        for (axis, delta) in origin.iter_mut().zip(deltas.iter()) {
            let current = axis
                .as_f64()
                .ok_or_else(|| anyhow::anyhow!("an 'origin' value is not a number"))?;
            *axis = serde_yaml::Value::Number((current + delta).into());
        }
    }
    Ok(serde_yaml::to_string(&value)?)
}

/// Open a file picker for the moving system's positions YAML, apply the shift, and save the result.
fn apply_shift_to_file(moving_kind: SystemKind, shift: XyzValue, ui_weak: &slint::Weak<AppWindow>) {
    let open_path = match rfd::FileDialog::new()
        .add_filter("YAML", &["yaml", "yml"])
        .set_title(format!("Open {} positions YAML", moving_kind.label()))
        .pick_file()
    {
        Some(p) => p,
        None => return,
    };

    let content = match std::fs::read_to_string(&open_path) {
        Ok(c) => c,
        Err(e) => {
            ui_set(ui_weak, move |ui| {
                ui.set_error_text(format!("Failed to read file: {e}").into());
            });
            return;
        }
    };

    let shifted = match moving_kind {
        SystemKind::Loco => shift_anchor_positions_yaml(&content, shift),
        SystemKind::Lighthouse => shift_lh_geometry_yaml(&content, shift),
    };

    let shifted_yaml = match shifted {
        Ok(y) => y,
        Err(e) => {
            ui_set(ui_weak, move |ui| {
                ui.set_error_text(format!("Failed to apply shift: {e}").into());
            });
            return;
        }
    };

    let stem = open_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("positions");
    let default_name = format!("{stem}_shifted.yaml");

    if let Some(save_path) = rfd::FileDialog::new()
        .add_filter("YAML", &["yaml", "yml"])
        .set_file_name(&default_name)
        .set_title(format!("Save shifted {} positions YAML", moving_kind.label()))
        .save_file()
    {
        if let Err(e) = std::fs::write(&save_path, shifted_yaml) {
            ui_set(ui_weak, move |ui| {
                ui.set_error_text(format!("Failed to save file: {e}").into());
            });
        } else {
            ui_set(ui_weak, move |ui| {
                ui.set_status_text("Shifted positions YAML saved.".into());
            });
        }
    }
}

// ── Command bus ───────────────────────────────────────────────────────────────

enum AppCommand {
    Connect(u64),
    Disconnect,
    Start(MeasurementConfig),
    Stop,
    Capture,
    Save,
    ApplyShift,
    SetLighthouseFwdToEstimator(bool),
    SetLocoFwdToEstimator(bool),
}

// ── UI update helpers ─────────────────────────────────────────────────────────

fn ui_set(ui_weak: &slint::Weak<AppWindow>, f: impl FnOnce(AppWindow) + Send + 'static) {
    let ui_weak = ui_weak.clone();
    slint::invoke_from_event_loop(move || {
        if let Some(ui) = ui_weak.upgrade() {
            f(ui);
        }
    })
    .ok();
}

// ── Measurement logic ─────────────────────────────────────────────────────────

/// Timeout applied to individual Crazyflie protocol round-trips (param set,
/// log block create/start). crazyflie-lib-rs's `wait_packet` has no internal
/// timeout, so a single dropped CRTP packet would otherwise hang forever.
const CF_OP_TIMEOUT: tokio::time::Duration = tokio::time::Duration::from_secs(5);

async fn with_timeout<T>(what: &str, fut: impl std::future::Future<Output = Result<T>>) -> Result<T> {
    match tokio::time::timeout(CF_OP_TIMEOUT, fut).await {
        Ok(result) => result,
        Err(_) => Err(anyhow::anyhow!("Timed out waiting for the Crazyflie to respond to: {what}")),
    }
}

/// Reads `num_samples` position samples from the continuous live position
/// broadcast stream (started on Connect and shared with the live view).
async fn read_position_stats(
    sample_rx: &mut broadcast::Receiver<(f32, f32, f32)>,
    num_samples: usize,
    stop_flag: &Arc<AtomicBool>,
    ui_weak: &slint::Weak<AppWindow>,
) -> Result<PositionStats> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();

    while xs.len() < num_samples {
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        let (x, y, z) = match tokio::time::timeout(CF_OP_TIMEOUT, sample_rx.recv()).await {
            Ok(Ok(sample)) => sample,
            Ok(Err(broadcast::error::RecvError::Lagged(_))) => continue,
            Ok(Err(broadcast::error::RecvError::Closed)) => {
                return Err(anyhow::anyhow!("Live position logging stopped"));
            }
            Err(_) => {
                if stop_flag.load(Ordering::Relaxed) {
                    break;
                }
                return Err(anyhow::anyhow!("Timed out waiting for log data from the Crazyflie"));
            }
        };

        xs.push(x);
        ys.push(y);
        zs.push(z);

        let count = xs.len();
        let total = num_samples;
        let ui_weak = ui_weak.clone();
        slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_live_x(format!("{:.3}", x).into());
                ui.set_live_y(format!("{:.3}", y).into());
                ui.set_live_z(format!("{:.3}", z).into());
                ui.set_progress_samples(count as i32);
                ui.set_total_samples(total as i32);
            }
        })
        .ok();
    }

    if xs.is_empty() {
        return Err(anyhow::anyhow!("No samples collected"));
    }

    Ok(PositionStats::from_vecs(xs, ys, zs))
}

async fn set_estimator_sources(cf: &crazyflie_lib::Crazyflie, lighthouse: u8, loco: u8) -> Result<()> {
    with_timeout("setting lighthouse.fwdToEstimator", async { Ok(cf.param.set("lighthouse.fwdToEstimator", lighthouse).await?) }).await?;
    with_timeout("setting loco.fwdToEstimator", async { Ok(cf.param.set("loco.fwdToEstimator", loco).await?) }).await?;
    Ok(())
}

async fn set_lighthouse_fwd_to_estimator(cf: &crazyflie_lib::Crazyflie, enabled: bool) -> Result<()> {
    with_timeout("setting lighthouse.fwdToEstimator", async {
        Ok(cf.param.set("lighthouse.fwdToEstimator", enabled as u8).await?)
    })
    .await
}

async fn set_loco_fwd_to_estimator(cf: &crazyflie_lib::Crazyflie, enabled: bool) -> Result<()> {
    with_timeout("setting loco.fwdToEstimator", async {
        Ok(cf.param.set("loco.fwdToEstimator", enabled as u8).await?)
    })
    .await
}

/// Reads the current lighthouse/loco `fwdToEstimator` params so the UI checkboxes
/// reflect the Crazyflie's actual configuration on connect.
async fn get_estimator_sources(cf: &crazyflie_lib::Crazyflie) -> Result<(bool, bool)> {
    let lighthouse: u8 = with_timeout("reading lighthouse.fwdToEstimator", async {
        Ok(cf.param.get("lighthouse.fwdToEstimator").await?)
    })
    .await?;
    let loco: u8 = with_timeout("reading loco.fwdToEstimator", async {
        Ok(cf.param.get("loco.fwdToEstimator").await?)
    })
    .await?;
    Ok((lighthouse != 0, loco != 0))
}

async fn reset_estimator(cf: &crazyflie_lib::Crazyflie) -> Result<()> {
    with_timeout("setting kalman.resetEstimation=1", async { Ok(cf.param.set("kalman.resetEstimation", 1u8).await?) }).await?;
    with_timeout("setting kalman.resetEstimation=0", async { Ok(cf.param.set("kalman.resetEstimation", 0u8).await?) }).await?;
    Ok(())
}

/// Builds the on-screen instruction shown before a capture step.
///
/// `is_first_step` controls whether the user is told to (re)place the Crazyflie.
/// When both the base and moving systems are Lighthouse, the second step instead
/// tells the user to swap which system's base stations are physically visible
/// without moving the Crazyflie.
fn capture_instruction(
    role: &str,
    kind: SystemKind,
    other_kind: SystemKind,
    point_n: usize,
    total: usize,
    is_first_step: bool,
) -> String {
    let lead = if is_first_step {
        format!("Place the Crazyflie at location {point_n}/{total}. ")
    } else if other_kind == SystemKind::Lighthouse && kind == SystemKind::Lighthouse {
        "Without moving the Crazyflie, ".to_string()
    } else {
        String::new()
    };

    match kind {
        SystemKind::Lighthouse if other_kind == SystemKind::Lighthouse => {
            format!("{lead}block the OTHER lighthouse system's base stations so only the {role} system's base stations are visible, then click 'Capture {role}'.")
        }
        SystemKind::Lighthouse => {
            format!("{lead}Ready to capture the {role} system (Lighthouse position). Click 'Capture {role}'.")
        }
        SystemKind::Loco => {
            format!("{lead}Ready to capture the {role} system (Loco position). Click 'Capture {role}'.")
        }
    }
}

/// Switches the estimator to `kind`, resets it, waits for it to settle, and collects samples.
async fn capture_position(
    cf: &crazyflie_lib::Crazyflie,
    sample_rx: &mut broadcast::Receiver<(f32, f32, f32)>,
    kind: SystemKind,
    role: &str,
    config: &MeasurementConfig,
    stop_flag: &Arc<AtomicBool>,
    ui_weak: &slint::Weak<AppWindow>,
) -> Result<PositionStats> {
    let (lighthouse, loco) = kind.estimator_sources();
    let mode_text = format!("Capturing {role} ({})", kind.label());
    let is_base = role == "Base";
    let total = config.num_samples;
    let settle_time_s = config.settle_time_s;

    ui_set(ui_weak, move |ui| {
        ui.set_is_waiting_for_capture(false);
        ui.set_is_measuring(true);
        ui.set_current_mode(mode_text.into());
        ui.set_current_mode_is_base(is_base);
        ui.set_progress_samples(0);
        ui.set_total_samples(total as i32);
        ui.set_status_text(
            format!("Resetting the estimator and waiting {settle_time_s:.1}s for the position estimate to stabilize…").into(),
        );
    });

    set_estimator_sources(cf, lighthouse, loco).await?;
    reset_estimator(cf).await?;

    tokio::time::sleep(tokio::time::Duration::from_secs_f64(config.settle_time_s)).await;

    // Discard samples buffered (or missed) before/during the settle period so
    // collection starts from fresh, post-settle data. A lagged receiver must
    // keep draining through `Lagged` errors too — otherwise the first
    // `try_recv` just consumes the lag marker and leaves a backlog of stale
    // samples that the next step would race through instantly.
    loop {
        match sample_rx.try_recv() {
            Ok(_) | Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(_) => break,
        }
    }

    ui_set(ui_weak, |ui| {
        ui.set_status_text("Collecting samples…".into());
    });

    read_position_stats(sample_rx, config.num_samples, stop_flag, ui_weak).await
}

async fn run_measurement(
    cf: Arc<crazyflie_lib::Crazyflie>,
    config: MeasurementConfig,
    ui_weak: slint::Weak<AppWindow>,
    mut measure_rx: mpsc::Receiver<()>,
    stop_flag: Arc<AtomicBool>,
    sample_tx: broadcast::Sender<(f32, f32, f32)>,
) -> Vec<PointResult> {
    let mut results: Vec<PointResult> = Vec::new();
    let mut sample_rx = sample_tx.subscribe();

    for point_idx in 0..config.num_points {
        let n = point_idx + 1;
        let total = config.num_points;

        // ── Step 1: capture the "base" system ───────────────────────
        {
            let instruction = capture_instruction("Base", config.base_kind, config.moving_kind, n, total, true);
            ui_set(&ui_weak, move |ui| {
                ui.set_is_waiting_for_capture(true);
                ui.set_is_measuring(false);
                ui.set_current_mode("".into());
                ui.set_current_point(n as i32);
                ui.set_total_points_display(total as i32);
                ui.set_capture_button_label("Capture Base".into());
                ui.set_status_text(instruction.into());
            });
        }

        match measure_rx.recv().await {
            Some(()) => {}
            None => break,
        }
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        let base_stats = match capture_position(&cf, &mut sample_rx, config.base_kind, "Base", &config, &stop_flag, &ui_weak).await {
            Ok(s) => s,
            Err(e) => {
                ui_set(&ui_weak, move |ui| {
                    ui.set_error_text(format!("Base capture failed: {e}").into());
                    ui.set_is_running(false);
                });
                set_estimator_sources(&cf, 1, 1).await.ok();
                return results;
            }
        };
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        // ── Step 2: capture the "moving" system ─────────────────────
        {
            let instruction = capture_instruction("Moving", config.moving_kind, config.base_kind, n, total, false);
            ui_set(&ui_weak, move |ui| {
                ui.set_is_waiting_for_capture(true);
                ui.set_is_measuring(false);
                ui.set_current_mode("".into());
                ui.set_capture_button_label("Capture Moving".into());
                ui.set_status_text(instruction.into());
            });
        }

        match measure_rx.recv().await {
            Some(()) => {}
            None => break,
        }
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        let moving_stats = match capture_position(&cf, &mut sample_rx, config.moving_kind, "Moving", &config, &stop_flag, &ui_weak).await {
            Ok(s) => s,
            Err(e) => {
                ui_set(&ui_weak, move |ui| {
                    ui.set_error_text(format!("Moving capture failed: {e}").into());
                    ui.set_is_running(false);
                });
                set_estimator_sources(&cf, 1, 1).await.ok();
                return results;
            }
        };
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        // ── Accumulate result ───────────────────────────────────────
        let point = PointResult { point_index: point_idx, base: base_stats, moving: moving_stats };

        {
            let dx = format!("{:+.3}", point.dx());
            let dy = format!("{:+.3}", point.dy());
            let dz = format!("{:+.3}", point.dz());
            let bx = format!("{:.3}", point.base.mean_x);
            let by = format!("{:.3}", point.base.mean_y);
            let bz = format!("{:.3}", point.base.mean_z);
            let mx = format!("{:.3}", point.moving.mean_x);
            let my = format!("{:.3}", point.moving.mean_y);
            let mz = format!("{:.3}", point.moving.mean_z);
            let pt_num = (point_idx + 1) as i32;

            // Per-system combined std dev for this point, regardless of which
            // role (base/moving) each system played.
            let lh_std = match (config.base_kind, config.moving_kind) {
                (SystemKind::Lighthouse, _) => Some(point.base.total_std()),
                (_, SystemKind::Lighthouse) => Some(point.moving.total_std()),
                _ => None,
            };
            let loco_std = match (config.base_kind, config.moving_kind) {
                (SystemKind::Loco, _) => Some(point.base.total_std()),
                (_, SystemKind::Loco) => Some(point.moving.total_std()),
                _ => None,
            };
            let lh_std = lh_std.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "--".to_string());
            let loco_std = loco_std.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "--".to_string());

            ui_set(&ui_weak, move |ui| {
                let row = PointResultData {
                    point: pt_num,
                    base_x: bx.into(), base_y: by.into(), base_z: bz.into(),
                    moving_x: mx.into(), moving_y: my.into(), moving_z: mz.into(),
                    dx: dx.into(), dy: dy.into(), dz: dz.into(),
                    lh_std: lh_std.into(), loco_std: loco_std.into(),
                };
                let model = ui.get_point_results();
                let mut rows: Vec<PointResultData> = (0..model.row_count())
                    .map(|i| model.row_data(i).unwrap())
                    .collect();
                rows.push(row);
                ui.set_point_results(slint::ModelRc::new(slint::VecModel::from(rows)));
            });
        }

        results.push(point);
    }

    // Restore both estimator sources
    set_estimator_sources(&cf, 1, 1).await.ok();

    results
}

/// Continuously streams position log data for the lifetime of the connection.
///
/// Started once on Connect and aborted on Disconnect. Each sample is broadcast
/// to any measurement run currently collecting samples, pushed into the
/// rolling `LIVE_WINDOW` buffer used for the live std-dev and 3D view, and
/// reflected in the live X/Y/Z readout.
async fn run_live_logging(
    cf: Arc<crazyflie_lib::Crazyflie>,
    log_period_ms: u64,
    live_state: Arc<Mutex<LiveState>>,
    sample_tx: broadcast::Sender<(f32, f32, f32)>,
    ui_weak: slint::Weak<AppWindow>,
) {
    let mut log_block = match with_timeout("creating live log block", async { Ok(cf.log.create_block().await?) }).await {
        Ok(b) => b,
        Err(e) => {
            ui_set(&ui_weak, move |ui| {
                ui.set_error_text(format!("Failed to start live logging: {e}").into());
            });
            return;
        }
    };

    let setup = async {
        log_block.add_variable("stateEstimate.x").await?;
        log_block.add_variable("stateEstimate.y").await?;
        log_block.add_variable("stateEstimate.z").await?;
        let period = crazyflie_lib::subsystems::log::LogPeriod::from_millis(log_period_ms)?;
        Ok(log_block.start(period).await?)
    };

    let log_stream = match with_timeout("starting live log block", setup).await {
        Ok(s) => s,
        Err(e) => {
            ui_set(&ui_weak, move |ui| {
                ui.set_error_text(format!("Failed to start live logging: {e}").into());
            });
            return;
        }
    };

    loop {
        let data = match log_stream.next().await {
            Ok(d) => d,
            Err(_) => break,
        };

        let x: f32 = data.data.get("stateEstimate.x").and_then(|v| (*v).try_into().ok()).unwrap_or(0.0);
        let y: f32 = data.data.get("stateEstimate.y").and_then(|v| (*v).try_into().ok()).unwrap_or(0.0);
        let z: f32 = data.data.get("stateEstimate.z").and_then(|v| (*v).try_into().ok()).unwrap_or(0.0);

        // Best-effort: no measurement may be subscribed right now.
        let _ = sample_tx.send((x, y, z));

        let (std_x, std_y, std_z) = {
            let mut state = live_state.lock().unwrap();
            push_live_sample(&mut state, LiveSample { t: Instant::now(), x, y, z });
            live_std(&state)
        };

        ui_set(&ui_weak, move |ui| {
            ui.set_live_x(format!("{:.3}", x).into());
            ui.set_live_y(format!("{:.3}", y).into());
            ui.set_live_z(format!("{:.3}", z).into());
            ui.set_live_std_x(format!("{:.3}", std_x).into());
            ui.set_live_std_y(format!("{:.3}", std_y).into());
            ui.set_live_std_z(format!("{:.3}", std_z).into());
        });
    }
}

// ── Async backend ─────────────────────────────────────────────────────────────

async fn async_backend(
    mut cmd_rx: mpsc::Receiver<AppCommand>,
    ui_weak: slint::Weak<AppWindow>,
    pending_yaml: Arc<Mutex<Option<(String, String)>>>,
    pending_shift: Arc<Mutex<Option<(SystemKind, XyzValue)>>>,
    sample_tx: broadcast::Sender<(f32, f32, f32)>,
    live_state: Arc<Mutex<LiveState>>,
) {
    let link_context = Arc::new(crazyflie_link::LinkContext::new());
    let toc_cache = FileTocCache::new();
    let mut cf: Option<Arc<crazyflie_lib::Crazyflie>> = None;

    // Active measurement synchronisation
    let measure_tx: Arc<Mutex<Option<mpsc::Sender<()>>>> = Arc::new(Mutex::new(None));
    let stop_flag = Arc::new(AtomicBool::new(false));

    // Continuous live-position logging task, started on Connect and aborted on Disconnect.
    let mut live_log_task: Option<tokio::task::JoinHandle<()>> = None;

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            AppCommand::Connect(log_period_ms) => {
                ui_set(&ui_weak, |ui| {
                    ui.set_connection_status_text("Connecting…".into());
                    ui.set_error_text("".into());
                });

                let uri = match link_context.scan([0xe7; 5]).await {
                    Ok(uris) => uris.into_iter().find(|u| u.starts_with("usb://")),
                    Err(_) => None,
                };

                let uri = match uri {
                    Some(uri) => uri,
                    None => {
                        ui_set(&ui_weak, |ui| {
                            ui.set_connection_status_text("Disconnected".into());
                            ui.set_error_text("No Crazyflie found over USB.".into());
                        });
                        continue;
                    }
                };

                // Let the USB device settle after the scan (which briefly opens
                // it to read descriptors) before re-opening it for the real link.
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

                match crazyflie_lib::Crazyflie::connect_from_uri(
                    link_context.as_ref(),
                    &uri,
                    toc_cache.clone(),
                )
                .await
                {
                    Ok(connected) => {
                        let connected_cf = Arc::new(connected);
                        cf = Some(connected_cf.clone());
                        ui_set(&ui_weak, |ui| {
                            ui.set_is_connected(true);
                            ui.set_connection_status_text("Connected".into());
                            ui.set_status_text("Ready. Configure and press Start Alignment.".into());
                        });

                        // Reflect the Crazyflie's current estimator source forwarding in the checkboxes.
                        if let Ok((lighthouse, loco)) = get_estimator_sources(&connected_cf).await {
                            ui_set(&ui_weak, move |ui| {
                                ui.set_fwd_lighthouse_to_estimator(lighthouse);
                                ui.set_fwd_loco_to_estimator(loco);
                            });
                        }

                        live_state.lock().unwrap().buffer.clear();
                        live_log_task = Some(tokio::spawn(run_live_logging(
                            connected_cf,
                            log_period_ms,
                            live_state.clone(),
                            sample_tx.clone(),
                            ui_weak.clone(),
                        )));
                    }
                    Err(e) => {
                        ui_set(&ui_weak, move |ui| {
                            ui.set_connection_status_text("Disconnected".into());
                            ui.set_error_text(format!("Connection failed: {e}").into());
                        });
                    }
                }
            }

            AppCommand::Disconnect => {
                cf = None;
                if let Some(handle) = live_log_task.take() {
                    handle.abort();
                }
                live_state.lock().unwrap().buffer.clear();
                ui_set(&ui_weak, |ui| {
                    ui.set_is_connected(false);
                    ui.set_is_running(false);
                    ui.set_is_waiting_for_capture(false);
                    ui.set_is_measuring(false);
                    ui.set_current_mode("".into());
                    ui.set_live_x("--".into());
                    ui.set_live_y("--".into());
                    ui.set_live_z("--".into());
                    ui.set_live_std_x("--".into());
                    ui.set_live_std_y("--".into());
                    ui.set_live_std_z("--".into());
                    ui.set_connection_status_text("Disconnected".into());
                    ui.set_status_text("Connect a Crazyflie via USB to begin.".into());
                });
            }

            AppCommand::Start(config) => {
                if let Some(connected_cf) = cf.clone() {
                    // Reset state
                    stop_flag.store(false, Ordering::Relaxed);
                    let (new_measure_tx, measure_rx) = mpsc::channel::<()>(1);
                    *measure_tx.lock().unwrap() = Some(new_measure_tx);
                    let total = config.num_points;
                    let base_label = config.base_kind.label().to_string();
                    let moving_label = config.moving_kind.label().to_string();

                    ui_set(&ui_weak, move |ui| {
                        ui.set_is_running(true);
                        ui.set_has_final_result(false);
                        ui.set_error_text("".into());
                        ui.set_total_points_display(total as i32);
                        ui.set_result_base_system(base_label.into());
                        ui.set_result_moving_system(moving_label.into());
                        ui.set_point_results(slint::ModelRc::new(slint::VecModel::from(vec![])));
                    });

                    let ui_weak2 = ui_weak.clone();
                    let measure_tx2 = measure_tx.clone();
                    let stop_flag2 = stop_flag.clone();
                    let pending_yaml2 = pending_yaml.clone();
                    let pending_shift2 = pending_shift.clone();
                    let base_kind = config.base_kind;
                    let moving_kind = config.moving_kind;
                    let sample_tx2 = sample_tx.clone();

                    tokio::spawn(async move {
                        let results = run_measurement(
                            connected_cf,
                            config,
                            ui_weak2.clone(),
                            measure_rx,
                            stop_flag2,
                            sample_tx2,
                        )
                        .await;

                        // Drop measure sender — no more points expected
                        *measure_tx2.lock().unwrap() = None;

                        if results.is_empty() {
                            ui_set(&ui_weak2, |ui| {
                                ui.set_is_running(false);
                                ui.set_is_waiting_for_capture(false);
                                ui.set_is_measuring(false);
                                ui.set_current_mode("".into());
                                ui.set_status_text("Stopped.".into());
                            });
                            return;
                        }

                        // Compute final result
                        let ar = build_alignment_result(&results, base_kind, moving_kind);
                        let yaml = serde_yaml::to_string(&ar).unwrap_or_default();
                        let file_stem = format!(
                            "alignment_{}_to_{}",
                            ar.moving_system.to_lowercase(),
                            ar.base_system.to_lowercase()
                        );
                        *pending_yaml2.lock().unwrap() = Some((format!("{file_stem}.yaml"), yaml));
                        *pending_shift2.lock().unwrap() = Some((moving_kind, ar.shift_vector));

                        let shift_x = format!("{:+.3}", ar.shift_vector.x);
                        let shift_y = format!("{:+.3}", ar.shift_vector.y);
                        let shift_z = format!("{:+.3}", ar.shift_vector.z);
                        let base_system = ar.base_system.clone();
                        let moving_system = ar.moving_system.clone();

                        ui_set(&ui_weak2, move |ui| {
                            ui.set_is_running(false);
                            ui.set_is_waiting_for_capture(false);
                            ui.set_is_measuring(false);
                            ui.set_current_mode("".into());
                            ui.set_has_final_result(true);
                            ui.set_shift_x(shift_x.into());
                            ui.set_shift_y(shift_y.into());
                            ui.set_shift_z(shift_z.into());
                            ui.set_result_base_system(base_system.into());
                            ui.set_result_moving_system(moving_system.into());
                            ui.set_status_text(
                                "Alignment complete. Save the result or run again.".into(),
                            );
                        });
                    });
                }
            }

            AppCommand::Stop => {
                stop_flag.store(true, Ordering::Relaxed);
                // Drop the measure sender so the measurement task's recv() returns None
                *measure_tx.lock().unwrap() = None;
            }

            AppCommand::Capture => {
                if let Ok(guard) = measure_tx.lock() {
                    if let Some(tx) = guard.as_ref() {
                        let _ = tx.try_send(());
                    }
                }
            }

            AppCommand::Save => {
                let pending = pending_yaml.lock().unwrap().clone();
                if let Some((file_name, yaml_str)) = pending {
                    tokio::task::spawn_blocking(move || {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("YAML", &["yaml", "yml"])
                            .set_file_name(&file_name)
                            .save_file()
                        {
                            if let Err(e) = std::fs::write(&path, &yaml_str) {
                                eprintln!("Failed to save: {e}");
                            }
                        }
                    })
                    .await
                    .ok();
                }
            }

            AppCommand::SetLighthouseFwdToEstimator(enabled) => {
                if let Some(connected_cf) = cf.clone() {
                    if let Err(e) = set_lighthouse_fwd_to_estimator(&connected_cf, enabled).await {
                        ui_set(&ui_weak, move |ui| {
                            ui.set_error_text(format!("Failed to set lighthouse.fwdToEstimator: {e}").into());
                            ui.set_fwd_lighthouse_to_estimator(!enabled);
                        });
                    }
                }
            }

            AppCommand::SetLocoFwdToEstimator(enabled) => {
                if let Some(connected_cf) = cf.clone() {
                    if let Err(e) = set_loco_fwd_to_estimator(&connected_cf, enabled).await {
                        ui_set(&ui_weak, move |ui| {
                            ui.set_error_text(format!("Failed to set loco.fwdToEstimator: {e}").into());
                            ui.set_fwd_loco_to_estimator(!enabled);
                        });
                    }
                }
            }

            AppCommand::ApplyShift => {
                let pending = *pending_shift.lock().unwrap();
                if let Some((moving_kind, shift)) = pending {
                    let ui_weak2 = ui_weak.clone();
                    tokio::task::spawn_blocking(move || {
                        apply_shift_to_file(moving_kind, shift, &ui_weak2);
                    })
                    .await
                    .ok();
                }
            }
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    let pending_yaml: Arc<Mutex<Option<(String, String)>>> = Arc::new(Mutex::new(None));
    let pending_shift: Arc<Mutex<Option<(SystemKind, XyzValue)>>> = Arc::new(Mutex::new(None));
    let (cmd_tx, cmd_rx) = mpsc::channel::<AppCommand>(32);

    // Continuous live-position broadcast + rolling 5s buffer (shared with the 3D view).
    let (sample_tx, _) = broadcast::channel::<(f32, f32, f32)>(256);
    let live_state: Arc<Mutex<LiveState>> = Arc::new(Mutex::new(LiveState::default()));

    let ui = AppWindow::new().expect("Slint window");

    // Wire callbacks
    let tx = cmd_tx.clone();
    let ui_weak_connect = ui.as_weak();
    ui.on_connect_clicked(move || {
        let ui = ui_weak_connect.upgrade().unwrap();
        let log_period_ms: u64 = ui.get_config_log_period().parse().unwrap_or(100u64).max(10);
        let _ = tx.try_send(AppCommand::Connect(log_period_ms));
    });

    let tx = cmd_tx.clone();
    ui.on_disconnect_clicked(move || { let _ = tx.try_send(AppCommand::Disconnect); });

    let tx = cmd_tx.clone();
    let ui_weak_start = ui.as_weak();
    ui.on_start_clicked(move || {
        let ui = ui_weak_start.upgrade().unwrap();
        let config = MeasurementConfig {
            base_kind: SystemKind::from_label(&ui.get_config_base_system()),
            moving_kind: SystemKind::from_label(&ui.get_config_moving_system()),
            num_points: ui.get_config_num_points().parse().unwrap_or(4).max(1),
            num_samples: ui.get_config_num_samples().parse().unwrap_or(100).max(1),
            settle_time_s: ui.get_config_settle_time().parse().unwrap_or(5.0_f64).max(0.0),
        };
        let _ = tx.try_send(AppCommand::Start(config));
    });

    let tx = cmd_tx.clone();
    ui.on_stop_clicked(move || { let _ = tx.try_send(AppCommand::Stop); });

    let tx = cmd_tx.clone();
    ui.on_capture_clicked(move || { let _ = tx.try_send(AppCommand::Capture); });

    let tx = cmd_tx.clone();
    ui.on_save_result_clicked(move || { let _ = tx.try_send(AppCommand::Save); });

    let tx = cmd_tx.clone();
    ui.on_apply_shift_clicked(move || { let _ = tx.try_send(AppCommand::ApplyShift); });

    let tx = cmd_tx.clone();
    ui.on_fwd_lighthouse_to_estimator_toggled(move |enabled| { let _ = tx.try_send(AppCommand::SetLighthouseFwdToEstimator(enabled)); });

    let tx = cmd_tx.clone();
    ui.on_fwd_loco_to_estimator_toggled(move |enabled| { let _ = tx.try_send(AppCommand::SetLocoFwdToEstimator(enabled)); });

    let ui_weak = ui.as_weak();
    rt.spawn(async_backend(cmd_rx, ui_weak, pending_yaml, pending_shift, sample_tx, live_state.clone()));

    // ── Live position 3D view ───────────────────────────────────────────────
    {
        let ui_weak = ui.as_weak();
        let live_state = live_state.clone();
        let mut scene_renderer: Option<renderer::Scene3DRenderer> = None;

        ui.window()
            .set_rendering_notifier(move |state, graphics_api| match state {
                slint::RenderingState::RenderingSetup => {
                    let context = match graphics_api {
                        slint::GraphicsAPI::NativeOpenGL { get_proc_address } => unsafe {
                            glow::Context::from_loader_function_cstr(|s| get_proc_address(s))
                        },
                        _ => return,
                    };
                    scene_renderer = Some(renderer::Scene3DRenderer::new(context));
                }
                slint::RenderingState::BeforeRendering => {
                    if let (Some(renderer), Some(app)) = (scene_renderer.as_mut(), ui_weak.upgrade()) {
                        let width = app.get_viz_width() as u32;
                        let height = app.get_viz_height() as u32;
                        if width == 0 || height == 0 {
                            return;
                        }

                        let (current, trail) = {
                            let state = live_state.lock().unwrap();
                            let current = state.buffer.back().map(|s| renderer::UnitPos {
                                x: s.x,
                                y: s.y,
                                z: s.z,
                                color: [0.20, 0.85, 0.95],
                            });
                            let mut trail = Vec::with_capacity(state.buffer.len() * 3);
                            for s in state.buffer.iter() {
                                trail.push(s.x);
                                trail.push(s.y);
                                trail.push(s.z);
                            }
                            (current, trail)
                        };

                        let texture = renderer.render(
                            width, height,
                            app.get_cam_yaw(), app.get_cam_pitch(), app.get_cam_distance(),
                            app.get_cam_pan_x(), app.get_cam_pan_y(),
                            VIZ_GRID_RANGE, current, &trail,
                        );
                        app.set_viz_texture(texture);

                        app.window().request_redraw();
                    }
                }
                slint::RenderingState::RenderingTeardown => {
                    drop(scene_renderer.take());
                }
                _ => {}
            })
            .expect("Unable to set rendering notifier");
    }

    ui.run().expect("Slint event loop");
}
