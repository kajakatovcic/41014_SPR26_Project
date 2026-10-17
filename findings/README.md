# Findings

Raw notes per marking criterion, with the test outputs and figures they quote. Every number comes from the final
system on `integration-branch`- the merged spline LQR controller, with the EKF, docking and
both robustness mechanisms.

| File | Criterion | Main sources |
|:--|:--|:--|
| [findings_A_B.txt](findings_A_B.txt) | A (model), B (sensing, calibration) | `test_model`, `test_perception`, `test_calibration`, `test_waffle_lidar` |
| [findings_C_loop.txt](findings_C_loop.txt) | C (end-to-end loop, EKF) | `main.py`, `test_estimation` |
| [findings_D_control.txt](findings_D_control.txt) | D (control, and integrating Anthony's work) | `test_control`, `main.py --controller ...` |
| [findings_E_robustness.txt](findings_E_robustness.txt) | E (robustness, comparison of controller versions) | `test_robustness` (spline and straight) |
| [findings_explore.txt](findings_explore.txt) | B, C, D, E for the **explore mission** (the default: no map, localise from a marker, lidar planning) | `test_navigation`, `test_estimation`, `test_robustness`, `main.py` |

The C, D and E files describe the **route mission** (known map, start pose and waypoints). It is kept as the baseline
(`main.py --mission route`), and its numbers did not change when the explore mission was added.

## outputs/
The console output of each run:

| Output | Command |
|:--|:--|
| `setup_waffle_sensors.txt` | `python setup_tests/test_waffle_sensors.py --headless` |
| `setup_waffle_lidar.txt` | `python setup_tests/test_waffle_lidar.py` |
| `calibration.txt` | `python tests/test_calibration.py` |
| `perception.txt` | `python tests/test_perception.py` |
| `model.txt` | `python tests/test_model.py` |
| `estimation.txt` | `python tests/test_estimation.py` |
| `control.txt` | `python tests/test_control.py` |
| `main_lqr.txt`, `main_dr.txt`, `main_straight.txt`, `main_simple.txt` | route mission: `python main.py --mission route`, then the same with `--estimator dr`, `--controller straight` and `--controller simple` |
| `robustness_lqr.txt`, `robustness_straight.txt` | route mission: `python tests/test_robustness.py --trials 10 --mission route`, then the same with `--controller straight` |
| `navigation.txt` | `python tests/test_navigation.py` (explore) |
| `estimation_explore.txt` | `python tests/test_estimation.py` (explore; `estimation.txt` is the route mission's, now `--mission route`) |
| `robustness_explore.txt` | `python tests/test_robustness.py --trials 10` (explore) |
| `main_explore.txt`, `main_explore_dr.txt`, `main_explore_straight.txt`, `main_obstacle_appears.txt` | `python main.py`, then the same with `--estimator dr`, `--controller straight` and `--obstacle-appears 20` |

Every test passed. For the robustness outputs, the 12 scenarios were run in two batches of six and then merged.
The seeds are the same as in a single run.

## figures/
The plot each command shows, saved with the same name as its output. `setup_waffle_sensors` has no plot.
