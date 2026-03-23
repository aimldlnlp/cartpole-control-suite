| suite | scenario | controller | runs | success_count | success_rate | handoff_rate | settling_time_median | overshoot_deg_median | steady_state_error_deg_median | control_effort_median | first_balance_time_median | min_abs_theta_deg_median | balance_fraction_median | invalid_rate | track_violation_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nominal | full_task_hanging | Feedback Linearization (PFL) | 5 | 5 | 1.0000 | 1.0000 | 2.2600 | 10.0108 | 0.0007 | 128.8684 | 2.0600 | 0.0000 | 0.8628 | 0.0000 | 0.0000 |
| nominal | full_task_hanging | LQR | 5 | 5 | 1.0000 | 1.0000 | 1.6800 | 10.0164 | 0.0000 | 128.9864 | 2.0600 | 0.0000 | 0.8628 | 0.0000 | 0.0000 |
| nominal | full_task_hanging | Sliding Mode Control (SMC) | 5 | 0 | 0.0000 | 1.0000 | nan | 9.9890 | 0.0010 | 129.0751 | 2.0600 | 0.0000 | 0.8628 | 0.0000 | 0.0000 |
| nominal | local_small_angle | Feedback Linearization (PFL) | 5 | 5 | 1.0000 | 1.0000 | 5.0000 | 10.0204 | 0.0694 | 7.3829 | 0.1500 | 0.0002 | 0.9813 | 0.0000 | 0.0000 |
| nominal | local_small_angle | LQR | 5 | 5 | 1.0000 | 1.0000 | 2.5200 | 9.9220 | 0.0007 | 11.5282 | 0.1500 | 0.0000 | 0.9813 | 0.0000 | 0.0000 |
| nominal | local_small_angle | Sliding Mode Control (SMC) | 5 | 0 | 0.0000 | 1.0000 | nan | 9.9220 | 0.0146 | 8.2229 | 0.1500 | 0.0002 | 0.9813 | 0.0000 | 0.0000 |
| stress | friction_and_damping | Feedback Linearization (PFL) | 5 | 5 | 1.0000 | 1.0000 | 5.1200 | 9.9575 | 0.0420 | 7.4438 | 0.1500 | 0.0001 | 0.9850 | 0.0000 | 0.0000 |
| stress | friction_and_damping | LQR | 5 | 5 | 1.0000 | 1.0000 | 2.4900 | 9.8665 | 0.0000 | 11.4158 | 0.1500 | 0.0000 | 0.9850 | 0.0000 | 0.0000 |
| stress | friction_and_damping | Sliding Mode Control (SMC) | 5 | 0 | 0.0000 | 1.0000 | nan | 9.8665 | 0.0118 | 8.0980 | 0.1500 | 0.0003 | 0.9850 | 0.0000 | 0.0000 |
| stress | impulse_disturbance | Feedback Linearization (PFL) | 5 | 5 | 1.0000 | 1.0000 | 3.3300 | 10.0204 | 0.0032 | 8.7307 | 0.1500 | 0.0000 | 0.9850 | 0.0000 | 0.0000 |
| stress | impulse_disturbance | LQR | 5 | 5 | 1.0000 | 1.0000 | 3.0000 | 9.9220 | 0.0003 | 12.7643 | 0.1500 | 0.0000 | 0.9850 | 0.0000 | 0.0000 |
| stress | impulse_disturbance | Sliding Mode Control (SMC) | 5 | 0 | 0.0000 | 1.0000 | nan | 9.9220 | 0.0138 | 9.2480 | 0.1500 | 0.0000 | 0.9850 | 0.0000 | 0.0000 |
| stress | large_angle_recovery | Feedback Linearization (PFL) | 5 | 5 | 1.0000 | 1.0000 | 5.0400 | 14.9326 | 0.0498 | 28.6877 | 0.5300 | 0.0002 | 0.9471 | 0.0000 | 0.0000 |
| stress | large_angle_recovery | LQR | 5 | 5 | 1.0000 | 1.0000 | 2.9800 | 16.5887 | 0.0000 | 34.4738 | 0.5300 | 0.0000 | 0.9471 | 0.0000 | 0.0000 |
| stress | large_angle_recovery | Sliding Mode Control (SMC) | 5 | 0 | 0.0000 | 1.0000 | nan | 8.5685 | 0.0221 | 21.9233 | 0.5300 | 0.0012 | 0.9471 | 0.0000 | 0.0000 |
| stress | measurement_noise | Feedback Linearization (PFL) | 5 | 5 | 1.0000 | 1.0000 | 5.2500 | 10.0116 | 0.1192 | 13.2153 | 0.1500 | 0.0009 | 0.9850 | 0.0000 | 0.0000 |
| stress | measurement_noise | LQR | 5 | 5 | 1.0000 | 1.0000 | 2.5500 | 9.9133 | 0.1165 | 22.1281 | 0.1500 | 0.0011 | 0.9850 | 0.0000 | 0.0000 |
| stress | measurement_noise | Sliding Mode Control (SMC) | 5 | 0 | 0.0000 | 1.0000 | nan | 9.9104 | 0.1142 | 18.4928 | 0.1500 | 0.0003 | 0.9850 | 0.0000 | 0.0000 |
| stress | parameter_mismatch | Feedback Linearization (PFL) | 5 | 5 | 1.0000 | 1.0000 | 4.7700 | 9.7914 | 0.0192 | 8.7910 | 0.1500 | 0.0001 | 0.9850 | 0.0000 | 0.0000 |
| stress | parameter_mismatch | LQR | 5 | 5 | 1.0000 | 1.0000 | 2.4700 | 9.6916 | 0.0000 | 14.2899 | 0.1500 | 0.0000 | 0.9850 | 0.0000 | 0.0000 |
| stress | parameter_mismatch | Sliding Mode Control (SMC) | 5 | 0 | 0.0000 | 1.0000 | nan | 9.6912 | 0.0092 | 9.4568 | 0.1500 | 0.0002 | 0.9850 | 0.0000 | 0.0000 |