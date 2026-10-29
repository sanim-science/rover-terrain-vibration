"""IMU model: gravity sign, body-frame rotation, filter/decimation shapes."""
import numpy as np
from scipy.spatial.transform import Rotation

from rover.sim import SimCfg, imu_from_states


def _states(n, quat, vel=None, angv=None):
    q = np.tile(quat, (n, 1))
    v = np.zeros((n, 3)) if vel is None else vel
    w = np.zeros((n, 3)) if angv is None else angv
    return v, w, q


def test_level_at_rest_reads_plus_g_up():
    cfg = SimCfg(noise=None)
    v, w, q = _states(2000, [0, 0, 0, 1])
    out = imu_from_states(v, w, q, cfg, np.random.default_rng(0))
    assert out["acc"].shape == (200, 3)
    assert np.allclose(out["acc"].mean(0), [0, 0, 9.81], atol=1e-6)


def test_nose_up_gives_positive_x_reading():
    # nose up by 10 deg (rotation about +y is nose DOWN in a z-up/x-forward frame, so use -10)
    cfg = SimCfg(noise=None)
    q = Rotation.from_euler("y", -10, degrees=True).as_quat()
    v, w, q = _states(2000, q)
    a = imu_from_states(v, w, q, cfg, np.random.default_rng(0))["acc"].mean(0)
    assert np.isclose(a[0], 9.81 * np.sin(np.radians(10)), atol=1e-6)  # accelerometer x reads +g sin(pitch)
    assert np.isclose(a[2], 9.81 * np.cos(np.radians(10)), atol=1e-6)


def test_forward_acceleration_and_yaw_rate_in_body_frame():
    cfg = SimCfg(noise=None)
    n = 3000
    yaw = np.radians(90)                      # car faces world +y
    q = Rotation.from_euler("z", yaw).as_quat()
    t = np.arange(n) / cfg.phys_hz
    vel = np.zeros((n, 3)); vel[:, 1] = 2.0 * t          # accelerating at 2 m/s^2 along world +y
    angv = np.zeros((n, 3)); angv[:, 2] = 0.5
    out = imu_from_states(vel, angv, np.tile(q, (n, 1)), cfg, np.random.default_rng(0))
    a, g = out["acc"][50:].mean(0), out["gyro"][50:].mean(0)
    assert np.allclose(a, [2.0, 0, 9.81], atol=1e-3)     # forward accel on body x
    assert np.allclose(g, [0, 0, 0.5], atol=1e-6)


if __name__ == "__main__":
    for f in [test_level_at_rest_reads_plus_g_up, test_nose_up_gives_positive_x_reading,
              test_forward_acceleration_and_yaw_rate_in_body_frame]:
        f(); print("ok", f.__name__)
