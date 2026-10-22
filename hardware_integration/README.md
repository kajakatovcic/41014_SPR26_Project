# Hardware integration

`run_hardware.py` runs the project's **explore mission** on the real TurtleBot3 Waffle Pi through rosbridge, using
the UTS-RI API in [`external/turtlebot_rosbridge`](../external/turtlebot_rosbridge).
Follow the main README's installation instructions in Setup step 3. 

The perception, estimation and control code is the same as in simulation:
- **Perception:** LiDAR walls and obstacles, ArUco range and bearing.
- **Estimation:** `MarkerSLAM`, localising from the reference marker with the wall-line heading.
- **Control:** `Navigator` with LiDAR occupancy grid + A* + spline, LQR path following, Week 9 docking, find and
  back off.

The only new implementation is the robot layer around it: 

| File | Purpose |
|:--|:--|
| `hw_config.py` | The real robot and room. Marker size, mounting height, IDs and dictionary; the measured dock offset; sensor mounts; speed limits; wheel joint names. Values marked **MEASURE** are simulation placeholders. |
| `robot_io.py` | `Waffle`: the API's `RosbridgeRobot` plus a `/odom` subscription (through its own `_add_subscriber`. The package itself is not modified). `RobotIO`: the conversions described below. |
| `run_hardware.py` | The mission loop at 20Hz on the wall clock, with dry run, live camera window, video and run log. |
| `check_sensors.py` | Checks every topic's rate and format, the wheel joint signs, the scan's angle convention, markers in view. |
| `calibrate_camera.py` | Camera calibration from the robot's own camera (Week 3 B/C, with lens distortion). Writes `camera_calibration.json`. |
| `sim_robot.py` | The same interface as `RosbridgeRobot`, backed by MuJoCo and answering in the real robot's message formats. Use `--sim` on any script to test the whole pipeline without the robot. |

### What `RobotIO` converts

| Robot topic | Becomes |
|:--|:--|
| `/joint_states` wheel angles | measured `(v, ω)`, plus the interval between the two readings from their time stamps (what `EncoderOdometry` gives in simulation), or `/odom`'s twist with `--odom-source odom` |
| `/scan` (LDS-01, "no return" = 0) | ranges on the project's 360 beams, invalid → ∞, as `perception.process_scan` expects |
| camera (BGR from the API) | undistorted with the calibration, converted to RGB, as expected by `perception.detect_markers` |
| `(v, ω)` | clipped to `V_LIMIT` / `W_LIMIT` and sent to `/cmd_vel` |

A watchdog stops the robot when odometry or LiDAR data is more than 0.5sec old. The TurtleBot firmware runs the wheel
speed loops itself, so the simulation's MuJoCo deadband compensation is not used here.

## Before the first run on the robot

1. **Connect.** Power the robot on - its service starts bringup, the camera and rosbridge. Join the `turtlebot-MMR`
   Wi-Fi and `ping 192.168.1.2XX` (`XX` = TurtleBot ID).
2. **Measure the room and fill in `hw_config.py`:**
   - the printed markers' black-square side (`MARKER_SIDE`), centre height (`MARKER_HEIGHT`) and dictionary
     (`ARUCO_DICT`);
   - which marker is the reference (`REF_MARKER_ID`). It must hang on a straight wall at least 1m long so the
     LiDAR can give its facing direction.
   - `DOCK_OFFSET`: from the reference marker's centre, how far out from its wall (x) and along it to the left (y)
     the dock marker's centre is, and which way the dock marker faces (π if it faces back towards the reference
     wall);
   - the camera and LiDAR positions on the robot (`CAMERA_POS`, `T_RS`).
3. **Check the sensors (no motion).**
   ```
   python hardware_integration/check_sensors.py 192.168.1.2XX
   ```
   - While it runs, drive the robot forward a little with the API's teleop in a second terminal:
     `python external/turtlebot_rosbridge/examples/teleop.py 192.168.1.2XX`. Both wheels should turn positive. If
     not, it prints the `WHEEL_SIGNS` to use.
   - Put an object straight in front of the robot: the nearest return should read about 0°.
   - Point the camera at a marker: its ID, range and height should be right.
4. **Calibrate the camera.** You need a printed checkerboard. Measure its internal corners and square size.
   ```
   python hardware_integration/calibrate_camera.py 192.168.1.2XX --grid 9 6 --square 0.04
   ```
   Keep 15–25 views across the whole image, at different distances and tilts. Aim for an RMS under 0.5px.
5. **Dry run.** The robot never moves.
   ```
   python hardware_integration/run_hardware.py 192.168.1.2XX --dry-run --show-camera
   ```
   Face the robot at the reference marker. It should localise within about 1sec (the summary says "marker yaw from
   the wall") and plan a path to the dock. The plot shows its LiDAR map and plan in the reference marker's frame.
   Drive it around with teleop: the estimated path should follow.
6. **Full run.** Keep a hand on Ctrl+C which stops the robot. Speeds are capped at 0.15m/s and 1rad/s.
   ```
   python hardware_integration/run_hardware.py 192.168.1.2XX --show-camera --video runs/run1.mp4 --log runs/run1.pkl
   ```

To test any of this without the robot, add `--sim`. For example `python hardware_integration/run_hardware.py --sim`
docks in about 52sec (8mm, −2mm lateral), and `--obstacle-appears 20` makes it replan around an obstacle.

## Differences from the simulation to expect

- **Timing:** The loop must keep up in real time.
  - Processing a LiDAR scan dominates. RANSAC uses 150 hypotheses per wall here (500 in simulation). On the
    simulated stand-in, loops took 21ms on average, 60ms at the 95th percentile and 96ms at worst, against a
    50ms budget.
  - The summary reports these numbers for every run, so watch them on the robot's laptop.
- **Sensor timing.**
  - The scan and the camera frame are not taken at the same instant, and one scan takes 0.2sec to sweep. The
    wall-line heading update is therefore only used while the robot turns slower than 0.2rad/s.
  - Fast turns smear the scan.
- **Noise models are the simulation's.** The camera's range and bearing noise (`perception.SIGMA_*`) and the
  odometry noise were fitted in simulation. If the EKF rejects many measurements (summary: gate rejections) or
  docks off-centre, refit them from real data. The run log keeps every observation for offline replay.
- **`BIAS_PX = 0`.** The 0.64px corner bias was a property of the simulated renderer.
- **Things the simulation did not have:** lighting, motion blur, reflective or black surfaces the LiDAR misses, and
  people moving.
- **Accuracy depends on `DOCK_OFFSET`.** An error in it moves the pre-dock pose by the same amount. Docking itself
  steers on the camera, so it still centres on the dock marker.
