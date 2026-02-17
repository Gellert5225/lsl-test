"""IMU orientation utilities — quaternion → Euler / rotation matrix.

Provides conversions needed by the renderer to pose the digital twin's
skeleton from IMU quaternion data (qw, qx, qy, qz).
"""

from __future__ import annotations

import numpy as np


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) to Euler angles (roll, pitch, yaw) in radians.

    Parameters
    ----------
    q : ndarray, shape (4,) or (N, 4)
        Quaternion(s) in [qw, qx, qy, qz] order.

    Returns
    -------
    ndarray, shape (3,) or (N, 3)
        Euler angles [roll, pitch, yaw] in radians.
    """
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[np.newaxis, :]

    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    result = np.stack([roll, pitch, yaw], axis=-1)
    return result[0] if single else result


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion to a 3×3 rotation matrix.

    Parameters
    ----------
    q : ndarray, shape (4,)
        Quaternion in [qw, qx, qy, qz] order (must be unit length).

    Returns
    -------
    ndarray, shape (3, 3)
        Rotation matrix.
    """
    q = np.asarray(q, dtype=np.float64)
    qw, qx, qy, qz = q

    return np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),       1 - 2*(qx*qx + qy*qy)],
    ])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion(s) to unit length.

    Parameters
    ----------
    q : ndarray, shape (4,) or (N, 4)

    Returns
    -------
    ndarray — same shape, unit length.
    """
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / (norm + 1e-12)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions.

    Parameters
    ----------
    q0, q1 : ndarray, shape (4,)
        Unit quaternions in [qw, qx, qy, qz] order.
    t : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    ndarray, shape (4,) — interpolated unit quaternion.
    """
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    dot = np.dot(q0, q1)

    # Ensure shortest path
    if dot < 0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    # If very close, use linear interpolation to avoid numerical issues
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return quat_normalize(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    s0 = np.sin((1 - t) * theta) / sin_theta
    s1 = np.sin(t * theta) / sin_theta

    return quat_normalize(s0 * q0 + s1 * q1)


def extract_quaternion(imu_sample: np.ndarray) -> np.ndarray:
    """Extract the quaternion portion from a 7-channel IMU sample.

    Parameters
    ----------
    imu_sample : ndarray, shape (7,)
        [qw, qx, qy, qz, ax, ay, az]

    Returns
    -------
    ndarray, shape (4,) — [qw, qx, qy, qz]
    """
    return np.asarray(imu_sample[:4], dtype=np.float64)


def extract_acceleration(imu_sample: np.ndarray) -> np.ndarray:
    """Extract the acceleration portion from a 7-channel IMU sample.

    Parameters
    ----------
    imu_sample : ndarray, shape (7,)
        [qw, qx, qy, qz, ax, ay, az]

    Returns
    -------
    ndarray, shape (3,) — [ax, ay, az] in m/s²
    """
    return np.asarray(imu_sample[4:7], dtype=np.float64)
