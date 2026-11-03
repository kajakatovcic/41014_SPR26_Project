# Set-Up Tests

This folder houses all python scripts to help you test if you have set up your virtual environment correctly,
as well as your Mujoco installation + all required packages.

Install everything first by following **Setup** in the [main README](../README.md) (steps 1 and 2), then run these
scripts from the repository root with the virtual environment activated. Every package they need is installed there.
In particular, OpenCV comes from `opencv-contrib-python==4.10.0.84`: do not install `opencv-python` as well, because
both write the same `cv2` folder and the second one overwrites the first.

* ### MuJoCo Sim
Use `test_sim.py` to ensure you have correctly installed MuJoCo. It opens an empty MuJoCo viewer; close the window to
finish.
```
python setup_tests/test_sim.py
```
* ### Robotis MuJoCo Menagerie
Use `test_waffle.py` to test whether you have correctly installed MuJoCo and the `robotis_mujoco_menagerie` package
(the git submodule). It drives the Waffle Pi straight for 5 s, then spins it in place, in the viewer.
```
python setup_tests/test_waffle.py
```
* ### TurtleBot3 Waffle Pi Sensors
Use `test_waffle_sensors.py` to test the TurtleBot3 Waffle Pi with a 2D lidar (360 beams, 0.12-3.5 m) and an RGB camera (640x480).
The robot wanders around a walled arena avoiding obstacles with the lidar. The MuJoCo viewer shows the lidar beams, and an
OpenCV window shows the camera image next to a top-down lidar scan.
```
python setup_tests/test_waffle_sensors.py               # 60 s with viewer + camera/lidar window
python setup_tests/test_waffle_sensors.py --headless    # no windows, prints sensor stats
```
* ### TurtleBot3 Waffle Pi Lidar
Use `test_waffle_lidar.py` to check the Waffle Pi's 2D lidar against ground truth using tutorial methods.
This is a good base line for what we should be using within our assignment.
Prints PASS/FAIL for each part, then shows a summary figure.
  1. Range-bearing model, SE3 sensor/robot frames and 1 cm range noise (Week 4, Part 5), checked against an analytic ray cast of the arena (Week 4, Part 2).
  2. Voxel downsampling and outlier removal (Week 4), RANSAC wall fitting and DBSCAN obstacle clustering (Week 4, Part 4), compared with the true walls and obstacles.
  3. The robot drives a unicycle circle (Week 7). Point-to-plane ICP between consecutive scans (Week 4) is chained into lidar odometry and compared with ground truth and wheel-encoder dead reckoning.
```
python setup_tests/test_waffle_lidar.py                 # run all parts then show the figure
python setup_tests/test_waffle_lidar.py --sigma 0.02    # try a noisier lidar
python setup_tests/test_waffle_lidar.py --no-plot       # results only
```
