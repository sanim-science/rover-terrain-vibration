"""Drive PyBullet's racecar over a generated heightfield and log a simulated IMU.

Coordinate conventions
----------------------
World: x along the track, y to the left, z up.  Car body frame: x forward,
y left, z up (the racecar URDF already uses this).

IMU model
---------
The IMU sits at the chassis centre of mass (the `chassis_inertia` link).
  accelerometer  f_b = R^T (a_w - g_w),  g_w = (0, 0, -9.81)
                 -> a car at rest reads (0, 0, +9.81)
  gyroscope      w_b = R^T w_w
a_w is the finite difference of the link's world velocity at the physics rate.
Both are low-pass filtered (4th-order Butterworth, causal, like a MEMS digital
low-pass filter) and decimated to the output rate, then per-run bias and white
noise are added.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field, asdict

import numpy as np
import pybullet as pb
import pybullet_data
from pybullet_utils import bullet_client
from scipy.signal import butter, sosfilt
from scipy.spatial.transform import Rotation

from .terrain import DX, FLAT_LEAD_IN, make_terrain, highpass_along_x

G = 9.81
WHEEL_R = 0.05
TRACK_HALF = 0.10   # wheel lateral offset from centreline [m]
WHEELBASE = 0.325


@dataclass
class NoiseCfg:
    acc_white: float = 0.02     # m/s^2, per output sample (std)
    gyro_white: float = 0.003   # rad/s
    acc_bias: float = 0.03      # m/s^2, per-run constant, std
    gyro_bias: float = 0.003    # rad/s
    wheel_rel: float = 0.005    # relative noise on wheel-encoder speed


@dataclass
class SimCfg:
    phys_hz: int = 1000             # physics rate; 100-400 Hz leaves contact chatter (checks.py)
    out_hz: int = 100
    duration: float = 20.0          # seconds simulated per run
    width: float = 1.2              # track width [m]
    motor_force: float = 10.0       # N*m per rear wheel (URDF effort limit)
    friction: float = 1.0
    tire_stiffness: float | None = None    # N/m; None = rigid contact. Soft contact chatters on heightfields
    tire_damping: float = 75.0              # N*s/m
    wheel_shape: str = "sphere"             # "sphere" (default) or "cylinder" (stock URDF)
    ground: str = "heightfield"             # "plane" only for noise-floor checks
    lowpass_hz: float = 40.0
    noise: NoiseCfg | None = field(default_factory=NoiseCfg)
    window_s: float = 1.0
    hop_s: float = 0.5
    settle_tol: float = 0.05        # +-5 % of steady speed
    settle_hold_s: float = 1.0
    settle_margin_s: float = 0.5


def racecar_urdf(wheel_shape: str) -> str:
    """Path to PyBullet's racecar URDF, optionally with sphere wheel collision shapes.

    The stock model collides through 4.5 cm-wide cylinders.  Bullet's cylinder-vs-
    triangle contact chatters (about 1 m/s^2 of fake vertical vibration on perfectly
    flat ground, see checks.py), so by default we swap only the wheel *collision*
    geometry for spheres of the same 5 cm radius.  Mass, inertia, joints and visuals
    are untouched."""
    src = os.path.join(pybullet_data.getDataPath(), "racecar", "racecar.urdf")
    if wheel_shape == "cylinder":
        return src
    out = os.path.join(tempfile.gettempdir(), f"racecar_{wheel_shape}_wheels.urdf")
    if not os.path.exists(out):
        txt = open(src).read()
        txt = txt.replace('<cylinder length="0.045" radius="0.05"/>', '<sphere radius="0.05"/>')
        mesh_dir = os.path.join(os.path.dirname(src), "meshes")
        txt = txt.replace('filename="meshes/', f'filename="{mesh_dir}/')
        with open(out, "w") as f:
            f.write(txt)
    return out


class RoverSim:
    """One physics client per process; call run() repeatedly."""

    def __init__(self):
        self.p = bullet_client.BulletClient(connection_mode=pb.DIRECT)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.cid = self.p._client

    # ------------------------------------------------------------------ setup
    def _build_world(self, h, cfg: SimCfg):
        p = self.p
        p.resetSimulation()
        p.setGravity(0, 0, -G)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0 / cfg.phys_hz, numSubSteps=0)
        nx, ny = h.shape
        if cfg.ground == "plane":       # analytic infinite plane, for noise-floor checks only
            ground = p.loadURDF("plane.urdf")
        else:
            z_off = 0.5 * (h.max() + h.min())
            # Verified in tests/test_heightfield_axes.py: passing h.T.ravel() with
            # numHeightfieldRows = nx maps h[ix, iy] to world (ix*DX, (iy-(ny-1)/2)*DX).
            shape = p.createCollisionShape(
                shapeType=pb.GEOM_HEIGHTFIELD, meshScale=[DX, DX, 1.0],
                heightfieldData=h.T.ravel().tolist(),
                numHeightfieldRows=nx, numHeightfieldColumns=ny)
            ground = p.createMultiBody(0, shape, basePosition=[(nx - 1) * DX / 2, 0.0, z_off])
        p.changeDynamics(ground, -1, lateralFriction=cfg.friction)

        car = p.loadURDF(racecar_urdf(cfg.wheel_shape), [0.8, 0.0, 0.005], useFixedBase=False)
        names = {p.getJointInfo(car, j)[1].decode(): j for j in range(p.getNumJoints(car))}
        links = {p.getJointInfo(car, j)[12].decode(): j for j in range(p.getNumJoints(car))}
        self.car = car
        self.rear = [names["left_rear_wheel_joint"], names["right_rear_wheel_joint"]]
        self.front = [names["left_front_wheel_joint"], names["right_front_wheel_joint"]]
        self.steer = [names["left_steering_hinge_joint"], names["right_steering_hinge_joint"]]
        self.imu_link = links["chassis_inertia"]
        for j in self.rear + self.front:
            kw = dict(lateralFriction=cfg.friction, rollingFriction=0.0, spinningFriction=0.0)
            if cfg.tire_stiffness is not None:
                kw.update(contactStiffness=cfg.tire_stiffness, contactDamping=cfg.tire_damping)
            p.changeDynamics(car, j, **kw)
        for j in self.front:  # free-rolling front wheels: disable default joint motors
            p.setJointMotorControl2(car, j, pb.VELOCITY_CONTROL, targetVelocity=0, force=0)

    # -------------------------------------------------------------- one run
    def run(self, kind, severity, speed, seed, cfg: SimCfg | None = None):
        cfg = cfg or SimCfg()
        p = self.p
        rng = np.random.default_rng(seed)
        length = FLAT_LEAD_IN + 1.0 + speed * cfg.duration + 3.0
        h, tparams = make_terrain(kind, severity, length, cfg.width, rng)
        self._build_world(h, cfg)

        n = int(round(cfg.duration * cfg.phys_hz))
        vel = np.empty((n, 3)); angv = np.empty((n, 3)); quat = np.empty((n, 4))
        pos = np.empty((n, 3)); wheel = np.empty(n)
        w_cmd = speed / WHEEL_R
        offtrack = flipped = False
        car, cid, rear, steer, imu = self.car, self.cid, self.rear, self.steer, self.imu_link
        pb.setJointMotorControlArray(car, rear, pb.VELOCITY_CONTROL, targetVelocities=[w_cmd] * 2,
                                     forces=[cfg.motor_force] * 2, physicsClientId=cid)
        y_lim = cfg.width / 2 - 0.15
        for k in range(n):
            # heading + lateral-offset steering so the car follows the centreline
            bp, bq = pb.getBasePositionAndOrientation(car, physicsClientId=cid)
            qx, qy, qz, qw = bq
            yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            delta = min(max(-1.5 * bp[1] - 1.0 * yaw, -0.35), 0.35)
            pb.setJointMotorControlArray(car, steer, pb.POSITION_CONTROL, targetPositions=[delta] * 2,
                                         forces=[10, 10], physicsClientId=cid)
            pb.stepSimulation(physicsClientId=cid)
            ls = pb.getLinkState(car, imu, computeLinkVelocity=1, physicsClientId=cid)
            pos[k], quat[k], vel[k], angv[k] = ls[0], ls[1], ls[6], ls[7]
            js = pb.getJointStates(car, rear, physicsClientId=cid)
            wheel[k] = 0.5 * (js[0][1] + js[1][1]) * WHEEL_R
            q = ls[1]
            up_z = 1 - 2 * (q[0] * q[0] + q[1] * q[1])
            offtrack = abs(ls[0][1]) > y_lim
            flipped = up_z < 0.5
            if offtrack or flipped:
                n = k + 1
                vel, angv, quat, pos, wheel = vel[:n], angv[:n], quat[:n], pos[:n], wheel[:n]
                break

        sig = imu_from_states(vel, angv, quat, cfg, rng)
        dec = cfg.phys_hz // cfg.out_hz
        m = len(sig["acc"])
        R = Rotation.from_quat(quat[dec - 1::dec][:m])
        sig["speed_true"] = R.inv().apply(vel[dec - 1::dec][:m])[:, 0]  # forward speed, body frame
        sig["wheel_speed"] = wheel[: m * dec].reshape(m, dec).mean(1)
        if cfg.noise is not None:
            sig["wheel_speed"] = sig["wheel_speed"] * (1 + cfg.noise.wheel_rel * rng.standard_normal(m))
        sig["pos"] = pos[dec - 1::dec][:m]
        sig["rpy"] = R.as_euler("xyz")
        sig["t"] = np.arange(1, m + 1) / cfg.out_hz

        settle_idx, v_ss = find_settle(sig["speed_true"], cfg)
        windows = make_windows(sig, settle_idx, h, cfg)
        meta = dict(kind=kind, severity=float(severity), speed_cmd=float(speed), seed=int(seed),
                    terrain=tparams, settle_time_s=settle_idx / cfg.out_hz if settle_idx is not None else None,
                    steady_speed=v_ss, offtrack=offtrack, flipped=flipped,
                    n_samples=m, cfg=asdict(cfg))
        return sig, windows, meta, h


def imu_from_states(vel, angv, quat, cfg: SimCfg, rng):
    dt = 1.0 / cfg.phys_hz
    acc_w = np.vstack([np.zeros((1, 3)), np.diff(vel, axis=0) / dt])
    acc_w[0] = acc_w[1] if len(acc_w) > 1 else 0
    R = Rotation.from_quat(quat)
    f_b = R.inv().apply(acc_w + np.array([0, 0, G]))
    w_b = R.inv().apply(angv)
    if cfg.lowpass_hz and cfg.lowpass_hz < cfg.phys_hz / 2:
        sos = butter(4, cfg.lowpass_hz, fs=cfg.phys_hz, output="sos")
        # start the filter in steady state at the first sample (avoids a start-up transient)
        f_b = sosfilt(sos, f_b - f_b[0], axis=0) + f_b[0]
        w_b = sosfilt(sos, w_b - w_b[0], axis=0) + w_b[0]
    dec = cfg.phys_hz // cfg.out_hz
    f_b, w_b = f_b[dec - 1::dec], w_b[dec - 1::dec]
    if cfg.noise is not None:
        nz = cfg.noise
        f_b = f_b + nz.acc_bias * rng.standard_normal(3) + nz.acc_white * rng.standard_normal(f_b.shape)
        w_b = w_b + nz.gyro_bias * rng.standard_normal(3) + nz.gyro_white * rng.standard_normal(w_b.shape)
    return {"acc": f_b, "gyro": w_b}


def find_settle(v, cfg: SimCfg):
    """First index after which the 1 s moving-average speed stays within +-tol of the
    steady speed for settle_hold_s, plus a safety margin.  Returns (idx, v_ss)."""
    fs = cfg.out_hz
    w = fs  # 1 s moving average
    if len(v) < 4 * fs:
        return None, None
    ma = np.convolve(v, np.ones(w) / w, mode="valid")  # ma[i] = mean(v[i:i+w])
    v_ss = float(np.median(v[len(v) // 2:]))
    ok = np.abs(ma - v_ss) <= cfg.settle_tol * abs(v_ss)
    hold = int(cfg.settle_hold_s * fs)
    for i in range(len(ok) - hold):
        if ok[i:i + hold].all():
            return i + int(cfg.settle_margin_s * fs), v_ss
    return None, v_ss


def make_windows(sig, start, h, cfg: SimCfg):
    """1 s windows (50 % overlap) from `start`, each labelled with the ground-truth
    roughness under the wheel tracks: RMS of the terrain height after removing
    wavelengths > 1 m, over the stretch the car covered in that window."""
    if start is None:
        return []
    fs = cfg.out_hz
    W, H = int(cfg.window_s * fs), int(cfg.hop_s * fs)
    hp = highpass_along_x(h)
    nx, ny = h.shape
    y0 = -(ny - 1) * DX / 2
    out = []
    for a in range(start, len(sig["acc"]) - W + 1, H):
        b = a + W
        x0, x1 = sig["pos"][a, 0] - 0.15, sig["pos"][b - 1, 0] + 0.2  # imu is ~0.15 m ahead of rear axle
        yc = sig["pos"][a:b, 1].mean()
        i0, i1 = max(int(x0 / DX), 0), min(int(x1 / DX) + 1, nx)
        lanes = [int(round((yc + s * TRACK_HALF - y0) / DX)) for s in (-1, 1)]
        prof = np.concatenate([hp[i0:i1, j] for j in lanes if 0 <= j < ny])
        out.append(dict(start=a, end=b,
                        roughness_mm=float(np.sqrt(np.mean(prof ** 2)) * 1000),
                        speed_true=float(sig["speed_true"][a:b].mean()),
                        wheel_speed=float(sig["wheel_speed"][a:b].mean())))
    return out
