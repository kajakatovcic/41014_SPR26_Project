# 41014_SPR26_Project
Integrated Project for 41014 Sensors and Control for Mechatronic Systems

## Project: Track A, mobile robot: visual waypoint navigation
A TurtleBot3 Waffle Pi with a 2D LiDAR and an RGB camera, simulated in MuJoCo. The robot navigates a room with obstacles using the
LiDAR and visually docks to the ArUco-marked charging station.

### Requirements
- [ ] **State-space model** of the platform
- [ ] **EKF** fusing wheel-encoder odometry with range-and-bearing (or pose) measurements from the camera
- [ ] **State-feedback or LQR trajectory tracker**

> **Assignment constraint:** the controller must use the **estimated** state and never ground truth.
> Ground truth from MuJoCo (`data.qpos`, `data.xpos`, ...) may only be used to evaluate and plot performance.

### Code layout
| Path | Purpose |
|:--|:--|
| `main.py` | Integrates perception, estimation and control in the simulation loop |
| `src/config.py` | Shared parameters between subsystems (robot geometry, sensor mounts and frames, camera intrinsics, room and marker map) |
| `src/simulation/` | MuJoCo world (room, obstacles, charging station, ArUco markers) and noisy sensor drivers |
| `src/perception/` | LiDAR processing (obstacles, walls) + ArUco range-and-bearing measurements with their covariance |
| `src/estimation/` | EKF: encoder odometry prediction, camera range-and-bearing/pose correction |
| `src/control/` | State-space model and the state-feedback/LQR trajectory tracker |
| `tests/` | Evaluation of each module against ground truth (`test_calibration.py`, `test_perception.py`, `test_model.py`, `test_estimation.py`, `test_control.py`, `test_robustness.py`) |
| `findings/` | Test outputs, figures and written findings per marking criterion (see [findings/README.md](findings/README.md)) |
| `setup_tests/` | Scripts to check the environment, MuJoCo and the TurtleBot3 sensors (see [setup_tests/README.md](setup_tests/README.md)) |
| `robotis_mujoco_menagerie/` | ROBOTIS MuJoCo models (git submodule), TurtleBot3 in `robotis_tb3/` |

### Tutorial content used within this assignment 
| Requirement | Tutorial |
|:--|:--|
| Camera model, ArUco detection + pose | Week 3 (Part E) |
| LiDAR point clouds, RANSAC, clustering, ICP | Week 4 (Part 5 and Bonus) |
| State-space model + unicycle error model | Week 5 (Part D) |
| Pole placement + LQR | Week 6 (Part A) |
| EKF with range-and-bearing measurements | Week 7 (Part C) |
| LQG - closing the loop on the EKF estimate | Week 8 (Parts B and C) |
| PBVS docking to an ArUco tag | Week 9 (Part D) |

## Setup
1. Clone with the submodule:
   ```
   git clone --recurse-submodules https://github.com/kajakatovcic/41014_SPR26_Project.git
   ```
   If you have already cloned it, run `git submodule update --init`.
2. Create the virtual environment and install the packages (Python 3.12):
   ```
   python3.12 -m venv .venv
   .venv\Scripts\activate
   python -m pip install mujoco numpy matplotlib open3d spatialmath-python tqdm control sympy
   python -m pip install opencv-contrib-python==4.10.0.84
   python -m pip install --no-deps machinevision-toolbox-python
   python -m pip install mvtb-data ansitable pgraph-python
   ```
3. In VS Code, select the interpreter with `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`.
4. Run the set-up tests to check everything works:
   ```
   python setup_tests/test_waffle_sensors.py
   python setup_tests/test_waffle_lidar.py
   ```

## Perception
`src/perception/perception.py` works on sensor data only (lidar ranges, camera images)and never on simulator state.

- **Lidar** (Week 4): scan → points in the robot frame through the sensor mount `T_RS` → voxel downsampling →
  walls by sequential RANSAC line fitting → obstacles by DBSCAN clustering
- **Camera** (Weeks 3, 7, 9): ArUco detection and PnP pose with Machine Vision Toolbox → marker pose in the
  robot frame through the camera mount `T_RC` → range and bearing `z = [rho, alpha]` to each known marker, with
  covariance `R` from the Week 3 model (range sigma grows with range squared and bearing sigma is constant).
  Detections are checked against the marker map which rejects reflections in the floor.

- **Camera calibration** (Week 3 - B & C): `src/perception/calibration.py` calibrates from checkerboard
  images (9 x 6 internal corners, 40mm squares) and checks the per-image reprojection error. Perception uses the
  calibrated intrinsics in `src/config.py` (`CAMERA_CAL_F`, `CAMERA_CAL_PP`) and not the simulator's true values.

Both are evaluated against ground truth. Run the calibration first since perception uses its result:
```
python tests/test_calibration.py   # about 1min: calibration, repeatability, camera mount check
python tests/test_perception.py    # about 30sec: 500 random poses
```
`test_calibration.py` gives the calibrated intrinsics for `src/config.py`. The corner bias and noise values in
`perception.py` come from `test_perception.py`. Rerun both if the camera, markers or detector change.

## State-space model
`src/control/model.py` is the model of the Waffle Pi that is shared by the estimator and controller.

- **Kinematics:** wheel speeds ↔ (v, ω) for the 0.033m wheels and 0.288m track. Commands that would exceed
  the wheel limit are scaled down keeping the path curvature.
- **Wheel actuators:** each wheel turns `frictionloss / kv = 1.0rad/s` slower than commanded with a 0.11sec 
  lag (both derived from the MuJoCo model's parameters). Additionally turning resistance costs a constant
  0.02rad/s of turn rate. `unicycle_to_wheels()` compensates for both. The usable top speed is 0.227m/s.
- **Unicycle** `x = (x, y, θ)`, `u = (v, ω)`: the continuous model (Week 5 D) and the exact-arc discrete model
  with its Jacobian (Week 7 C.4) for the EKF
- **Tracking-error model** about a reference moving at (v_r, ω_r) - the 3-state model and the 2-state
  cross-track/heading model of Weeks 5 D and 8 B which is discretised by zero-order hold at `CONTROL_DT` = 0.05sec.

To check the derivation, controllability, observability, discretisation and the model against the MuJoCo robot:
```
python tests/test_model.py   # about 10sec
```

## Running the full loop
`main.py` runs the mission in `src/config.py` (`START_POSE`, `WAYPOINTS`): simulation → sensor drivers →
perception → estimator → controller → wheel commands. Control and estimation run at 20Hz, the camera at 10Hz and
the lidar at 5Hz. The estimator and controller only receive sensor data and the estimate. Ground truth is logged
for **EVALUATION ONLY**.
```
python main.py              # prints a summary and shows the map and estimation-error plots (headless)
python main.py --viewer     # watch it in the MuJoCo viewer in real time + the robot's camera in its own window
python main.py --record     # save a video (overview + camera feed) to videos/mission.mp4 (or --record PATH)
```
The MuJoCo viewer opens on an overview of the room with the lidar rays shown. The "Robot camera" window streams
the robot's own view: the same noisy frames perception receives, with each detected marker outlined and labelled
with its id, range and bearing, plus the controller mode (`src/simulation/video.py`; for demonstration only, no
ground truth is drawn). `--viewer --record` does both. The video plays in real time at the camera's 10 frames per
second.
The EKF drives the controller (to use dead reckoning instead: `--estimator dr`). The simulated encoders have
realistic odometry errors (`SIM_WHEEL_SCALE_ERROR`, `SIM_WHEEL_SLIP_SIGMA` in `src/config.py`) which are unknown to the robot's software. The mission controller (LQR tracking, then docking) drives the robot. `--controller simple` uses the Week 8 C.4 placeholder instead as a baseline.

## Estimation
`src/estimation/estimation.py` is an EKF on the pose (x, y, θ), built from the Week 7 C functions:

- **Predict:** the exact-arc unicycle model driven by the encoder odometry. Q_d comes from a wheel odometry error
  model (radius uncertainty + slip) mapped onto the pose through the motion model's Jacobian.
- **Update:** each ArUco range and bearing measurement against the marker map, with perception's covariance, the
  bearing innovation wrapped, and a chi-square gate on the innovation to reject outliers.
- **Tuning** (`EKF_Q_SCALE`, `EKF_R_SCALE` in `src/config.py`) chosen by NEES over repeated missions (Week 7 D.3 + D.4).

To evaluate it over repeated missions with random odometry errors (~2min):
```
python tests/test_estimation.py
```

## Control
`src/control/control.py` - `MissionController`, runs on the estimated pose in three modes:

- **Track:** LQR path following on the cross-track/heading error model (Weeks 5 D, 8 B). Gains come from
  `ct.dlqr` at the control period, with Q and R from Bryson's rule (Week 6 A.5: 50 mm, 10°, 0.5 rad/s),
  gain-scheduled on speed.
  - The path is a smooth cubic spline through the route, arriving at the pre-dock pose already facing the dock.
    A curvature feedforward `ω_r = v·κ` lets it drive the curves instead of only correcting.
  - The speed drops where the curvature ahead is high.
  - It stops for lidar obstacles in its path, and plans a detour if the path stays blocked for 1 s.
- **Turn:** at the start, at the corners of straight detour legs, before docking, and whenever the heading error
  is beyond 30° (outside the linear model), it turns on the spot with the Lyapunov heading law ω = −k e_θ
  (Week 6 B.6).
- **Dock:** at the pre-dock pose it runs the Week 9 law on the dock marker, `v = λv (z_M − d*)` and
  `ω = −λω atan2(x_M, z_M)`, until the camera is `DOCK_DISTANCE` = 0.25m from the marker.

**Merged contributions from `Anthony_Branch`:**

- The spline geometry and curvature feedforward (`Trajectory` in `src/control/trajectory.py`).
- The route in `src/config.py` `WAYPOINTS` (his `ROUTE`, ending at the pre-dock pose).
- The route clearance check (`mission.clearance`, Part 0 of the control test).

`MissionController(smooth=False)`, or `python main.py --controller straight`, gives the earlier behaviour for
comparison: straight segments, turning on the spot at each corner.

To evaluate the route clearance, the design, the trade-off curve, missions (spline vs straight vs the placeholder) and docking from offset starts (~6min):
```
python tests/test_control.py
```

## Robustness
`tests/test_robustness.py` runs the whole mission (perception → EKF → LQR → docking) many times under 12 scenarios
that change the world but NOT the robot software: 

- nominal
- wrong start pose (small, large)
- 3x odometry errors
- 4x camera noise
- a camera calibration error
- 3x lidar noise
- missing landmarks
- a sideways shove while driving the route, and one while docking
- an unexpected obstacle on the route
- a combination of the above

It scores every trial against ground truth (success = docked within 30mm/5° with no collision) and gives each
failure a cause. Missions run in parallel worker processes (120 missions take ~10min on 16 cores). 
`--controller straight` runs the earlier straight-segment controller for comparison:
```
python tests/test_robustness.py --trials 10
python tests/test_robustness.py --trials 3 --scenarios push unexpected_obstacle    # a subset
```

While running these trials, we implemented two robustness mechanisms from our findings:
- **EKF lock-out recovery:** When every marker measurement has been rejected for 0.5sec, the filter inflates its covariance.
- **LiDAR detour planning:** When the path stays blocked for 1sec, the controller plans a sidestep around the obstacle.

`main.run()` takes the scenario settings as keyword arguments so any of them can be run on their own.
