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
| `tests/` | Evaluation of each module against ground truth (e.g. `test_perception.py`) |
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
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install mujoco numpy matplotlib open3d spatialmath-python tqdm
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

This subsystem is evaluated against ground truth (about 30sec with 500 random poses):
```
python tests/test_perception.py
```
The corner bias and noise values in `perception.py` come from this script and must be rerun if the camera, markers or detector change.
