| suite | scenario | controller | runs | success_count | success_rate | handoff_rate | settling_time_median | overshoot_deg_median | steady_state_error_deg_median | control_effort_median | first_balance_time_median | min_abs_theta_deg_median | balance_fraction_median | invalid_rate | track_violation_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nominal | full_task_hanging | Feedback Linearization (PFL) | 5 | 5 | 1.0000 | 1.0000 | 2.2600 | 10.0108 | 0.0007 | 128.8684 | 2.0600 | 0.0000 | 0.8628 | 0.0000 | 0.0000 |
| nominal | full_task_hanging | LQR | 5 | 5 | 1.0000 | 1.0000 | 1.6800 | 10.0164 | 0.0000 | 128.9864 | 2.0600 | 0.0000 | 0.8628 | 0.0000 | 0.0000 |
| nominal | full_task_hanging | Sliding Mode Control (SMC) | 5 | 0 | 0.0000 | 1.0000 | nan | 38.5056 | 11.6581 | 130.3756 | 2.0600 | 0.0110 | 0.6157 | 0.0000 | 1.0000 |
| nominal | local_small_angle | Feedback Linearization (PFL) | 5 | 5 | 1.0000 | 1.0000 | 5.0000 | 10.0204 | 0.0694 | 7.3829 | 0.1500 | 0.0002 | 0.9813 | 0.0000 | 0.0000 |
| nominal | local_small_angle | LQR | 5 | 5 | 1.0000 | 1.0000 | 2.5200 | 9.9220 | 0.0007 | 11.5282 | 0.1500 | 0.0000 | 0.9813 | 0.0000 | 0.0000 |
| nominal | local_small_angle | Sliding Mode Control (SMC) | 5 | 0 | 0.0000 | 1.0000 | nan | 40.5146 | 12.0244 | 11.8389 | 0.1500 | 1.2222 | 0.9435 | 0.0000 | 1.0000 |
| stress | friction_and_damping | Feedback Linearization (PFL) | 5 | 5 | 1.0000 | 1.0000 | 5.1200 | 9.9575 | 0.0420 | 7.4438 | 0.1500 | 0.0001 | 0.9850 | 0.0000 | 0.0000 |
| stress | friction_and_damping | LQR | 5 | 5 | 1.0000 | 1.0000 | 2.4900 | 9.8665 | 0.0000 | 11.4158 | 0.1500 | 0.0000 | 0.9850 | 0.0000 | 0.0000 |
| stress | friction_and_damping | Sliding Mode Control (SMC) | 5 | 0 | 0.0000 | 1.0000 | nan | 41.1913 | 12.3332 | 14.0536 | 0.1500 | 1.2590 | 0.9391 | 0.0000 | 1.0000 |
| stress | impulse_disturbance | Feedback Linearization (PFL) | 5 | 5 | 1.0000 | 1.0000 | 3.3300 | 10.0204 | 0.0032 | 8.7307 | 0.1500 | 0.0000 | 0.9850 | 0.0000 | 0.0000 |
| stress | impulse_disturbance | LQR | 5 | 5 | 1.0000 | 1.0000 | 3.0000 | 9.9220 | 0.0003 | 12.7643 | 0.1500 | 0.0000 | 0.9850 | 0.0000 | 0.0000 |
| stress | impulse_disturbance | Sliding Mode Control (SMC) | 5 | 0 | 0.0000 | 1.0000 | nan | 40.5146 | 12.0244 | 11.8389 | 0.1500 | 1.2222 | 0.9435 | 0.0000 | 1.0000 |
| stress | large_angle_recovery | Feedback Linearization (PFL) | 5 | 5 | 1.0000 | 1.0000 | 5.0400 | 14.9326 | 0.0498 | 28.6877 | 0.5300 | 0.0002 | 0.9471 | 0.0000 | 0.0000 |
| stress | large_angle_recovery | LQR | 5 | 5 | 1.0000 | 1.0000 | 2.9800 | 16.5887 | 0.0000 | 34.4738 | 0.5300 | 0.0000 | 0.9471 | 0.0000 | 0.0000 |
| stress | large_angle_recovery | Sliding Mode Control (SMC) | 5 | 0 | 0.0000 | 1.0000 | nan | 179.8223 | 118.0430 | 381.0828 | 0.5300 | 0.0018 | 0.1089 | 0.0000 | 0.0000 |
| stress | measurement_noise | Feedback Linearization (PFL) | 5 | 5 | 1.0000 | 1.0000 | 5.2500 | 10.0116 | 0.1192 | 13.2153 | 0.1500 | 0.0009 | 0.9850 | 0.0000 | 0.0000 |
| stress | measurement_noise | LQR | 5 | 5 | 1.0000 | 1.0000 | 2.5500 | 9.9133 | 0.1165 | 22.1281 | 0.1500 | 0.0011 | 0.9850 | 0.0000 | 0.0000 |
| stress | measurement_noise | Sliding Mode Control (SMC) | 5 | 0 | 0.0000 | 1.0000 | nan | 51.1968 | 14.1095 | 24.8387 | 0.1500 | 1.0352 | 0.9298 | 0.0000 | 1.0000 |
| stress | parameter_mismatch | Feedback Linearization (PFL) | 5 | 5 | 1.0000 | 1.0000 | 4.7700 | 9.7914 | 0.0192 | 8.7910 | 0.1500 | 0.0001 | 0.9850 | 0.0000 | 0.0000 |
| stress | parameter_mismatch | LQR | 5 | 5 | 1.0000 | 1.0000 | 2.4700 | 9.6916 | 0.0000 | 14.2899 | 0.1500 | 0.0000 | 0.9850 | 0.0000 | 0.0000 |
| stress | parameter_mismatch | Sliding Mode Control (SMC) | 5 | 0 | 0.0000 | 1.0000 | nan | 35.9681 | 11.4837 | 11.1628 | 0.1500 | 1.1978 | 0.9472 | 0.0000 | 1.0000 |