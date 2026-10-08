# 41014_SPR26_Project
Integrated Project for 41014 Sensors and Control for Mechatronic Systems

## Project: Track A, mobile robot: visual waypoint navigation
A TurtleBot3 Waffle Pi with a 2D LiDAR and an RGB camera, simulated in MuJoCo. The robot navigates a room with obstacles using the
LiDAR and visually docks to the ArUco-marked charging station.

### Requirements
- [ ] **State-space model** of the platform
- [ ] **EKF** fusing wheel-encoder odometry with range-and-bearing (or pose) measurements from the camera
- [ ] **State-feedback or LQR trajectory tracker**

> **Constraint:** the controller must use the **estimated** state, never ground truth.
> Ground truth from MuJoCo (`data.qpos`, `data.xpos`, ...) may only be used to evaluate and plot performance.

### Code layout
| Path | Purpose |
|:--|:--|
| `main.py` | Integrates perception, estimation and control in the simulation loop |
| `src/perception/` | LiDAR processing (obstacles, walls) and ArUco detection of the charging station |
| `src/estimation/` | EKF: encoder odometry prediction, camera range-and-bearing / pose correction |
| `src/control/` | State-space model and the state-feedback / LQR trajectory tracker |
| `setup_tests/` | Scripts to check the environment, MuJoCo and the TurtleBot3 sensors (see [setup_tests/README.md](setup_tests/README.md)) |
| `robotis_mujoco_menagerie/` | ROBOTIS MuJoCo models (git submodule), TurtleBot3 in `robotis_tb3/` |

### Relevant tutorial content
| Requirement | Tutorial |
|:--|:--|
| Camera model, ArUco detection and pose | Week 3 Part E |
| LiDAR point clouds, RANSAC, clustering, ICP | Week 4 Part 5 and Bonus |
| State-space model, unicycle error model | Week 5 Part D |
| Pole placement, LQR | Week 6 Part A |
| EKF with range-and-bearing measurements | Week 7 Part C |
| LQG: closing the loop on the EKF estimate | Week 8 Parts B and C |
| PBVS docking to an ArUco tag | Week 9 Part D |

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
   python -m pip install mujoco numpy opencv-python matplotlib open3d spatialmath-python
   ```
3. In VS Code, select the interpreter with `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`.
4. Run the set-up tests to check everything works:
   ```
   python setup_tests/test_waffle_sensors.py
   python setup_tests/test_waffle_lidar.py
   ```
