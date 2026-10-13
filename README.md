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
| `tests/` | Evaluation of each module against ground truth (`test_calibration.py`, `test_perception.py`, `test_model.py`) |
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
`src/perception/perception.py` works on sensor data only (lidar ranges, camera images), never on simulator state.

- **Lidar** (Week 4): scan → points in the robot frame through the sensor mount `T_RS` → voxel downsampling →
  walls by sequential RANSAC line fitting → obstacles by DBSCAN clustering
- **Camera** (Weeks 3, 7, 9): ArUco detection and PnP pose with Machine Vision Toolbox → marker pose in the
  robot frame through the camera mount `T_RC` → range and bearing `z = [rho, alpha]` to each known marker, with
  covariance `R` from the Week 3 model (range sigma grows with range squared and bearing sigma is constant).
  Detections are checked against the marker map which rejects reflections in the floor.

- **Camera calibration** (Week 3 - B & C): `src/perception/calibration.py` calibrates from checkerboard
  images (9 x 6 internal corners, 40mm squares) and checks the per-image reprojection error. Perception uses the
  calibrated intrinsics in `src/config.py` (`CAMERA_CAL_F`, `CAMERA_CAL_PP`) and not the simulator's true values.

Both are evaluated against ground truth. Run the calibration first, since perception uses its result:
```
python tests/test_calibration.py   # about 1 min: calibration, repeatability, camera mount check
python tests/test_perception.py    # about 30 s: 500 random poses
```
`test_calibration.py` gives the calibrated intrinsics for `src/config.py`. The corner bias and noise values in
`perception.py` come from `test_perception.py`. Rerun both if the camera, markers or detector change.

## State-space model
`src/control/model.py` is the model of the Waffle Pi that the estimator and controller share.

- **Kinematics:** wheel speeds ↔ (v, ω) for the 0.033m wheels and 0.288m track. Commands that would exceed
  the wheel limit are scaled down keeping the path curvature.
- **Wheel actuators:** each wheel turns `frictionloss / kv = 1.0 rad/s` slower than commanded with a 0.11sec 
  lag (both derived from the MuJoCo model's parameters). Additionally turning resistance costs a constant
  0.02rad/s of turn rate. `unicycle_to_wheels()` compensates for both. The usable top speed is 0.227m/s.
- **Unicycle** `x = (x, y, θ)`, `u = (v, ω)`: the continuous model (Week 5 D) and the exact-arc discrete model
  with its Jacobian (Week 7 C.4) for the EKF
- **Tracking-error model** about a reference moving at (v_r, ω_r): the 3-state model and the 2-state
  cross-track/heading model of Weeks 5 D and 8 B, discretised by zero-order hold at `CONTROL_DT` = 0.05sec.

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
python main.py              # headless: prints a summary and shows the map and estimation-error plots
python main.py --viewer     # watch it in the MuJoCo viewer, in real time
```
The EKF drives the controller (`--estimator dr` uses dead reckoning instead). The simulated encoders have
realistic odometry errors (`SIM_WHEEL_SCALE_ERROR`, `SIM_WHEEL_SLIP_SIGMA` in `src/config.py`) which are unknown to the
robot's software. Current placeholder: `WaypointFollower` in `src/control/control.py` (to become the LQR tracker
and docking).

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
