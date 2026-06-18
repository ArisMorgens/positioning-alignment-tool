# Crazyflie Positioning Alignment Tool

A small desktop tool for aligning the Loco Positioning System and Lighthouse
positioning system on a Crazyflie, so both report positions in the same
coordinate frame.

It connects to a Crazyflie over USB, walks you through capturing the
position reported by each system at one or more physical points, and
computes the translation (shift) needed to bring the "moving" system's
coordinates in line with the "base" system's.

## Build & run

```sh
cargo run --release
```

Requires a Crazyflie connected via USB, with both the Loco and/or Lighthouse
decks attached.

## Usage

1. Connect to the Crazyflie via USB.
2. Choose which system is the "base" (stays fixed) and which is "moving"
   (will be shifted).
3. Start the alignment and follow the on-screen instructions, placing the
   Crazyflie at each measurement point and capturing both systems' readings.
4. Review the per-point results, then save the alignment report or apply
   the computed shift to the moving system's positions YAML.
