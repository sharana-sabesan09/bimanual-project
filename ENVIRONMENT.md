# Ball Balance Environment

**Goal:** Train a policy to keep a ball balanced on a tray held by the Unitree G1 robot's right arm.

## Robot Setup

- Uses `g1_fixed_base.xml` — the torso is welded to the world (no floating base), so training doesn't have to deal with the robot falling over.
- The robot has 29 joints total, split into groups:
  - **Legs (12) + Waist (3) + Left arm (7)** — frozen via PD control at a standing pose every step. The RL policy never touches these.
  - **Right arm (7)** — the only thing the policy controls.

## Tray & Ball

- A **30×22cm tray** is rigidly attached to `right_wrist_yaw_link` — it moves exactly with the wrist.
- A **4cm radius ball** (0.1kg) is spawned just above the tray centre on reset.

## Action Space

7 continuous values — position targets (radians) for the 7 right-arm joints:

```
shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
```

## Observation Space (23-dim flat vector)

| Slice | Content |
|---|---|
| `[0:7]` | Right arm joint positions |
| `[7:14]` | Right arm joint velocities |
| `[14:17]` | Ball world position (x, y, z) |
| `[17:20]` | Ball velocity |
| `[20:23]` | Tray centre world position (the goal) |

## Reward

```
reward = exp(-3 * XY_distance_ball_to_tray)   # proximity bonus
       - 0.1 * ball_speed                      # penalise erratic motion
       - 10  if ball falls >15cm below tray    # terminal fall penalty
```

Episode ends when the ball falls off (`done=True`).

## Simulation

Built on [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) with a CPU backend, 50Hz timestep (`dt=0.02`).

## Running

```bash
# Visualise the environment with all joints frozen (sanity check)
python ball_balance_env.py
```
