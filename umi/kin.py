from __future__ import annotations

from scipy.spatial.transform import Rotation as RotationSciPy
import numpy as np


class JointConf(np.ndarray):
    """
    A Joint configuration
    """

    def __new__(cls, arr: np.ndarray):
        assert arr.shape == (7,)
        return arr.view(cls)


class Pose(np.ndarray):
    """
    A homogeneous transformation matrix of SE(3).
    """

    def __new__(cls, arr: np.ndarray):
        assert isinstance(arr, np.ndarray), f"Array must be np.ndarray. Got {type(arr)}"
        assert arr.shape == (4, 4), f"Array has to be of shape (4, 4). Got {arr.shape}"
        assert (
            np.max(np.linalg.det(arr[:3, :3])) <= 1.0 + 1e-7
        )  # internal checks of rotation
        return arr.view(cls)

    # used for type checkers to ensure that the result of Pose @ Pose -> Pose
    def __matmul__(self: "Pose", other: "Pose") -> "Pose":  # type: ignore
        return np.matmul(self, other).view(Pose)

    @property
    def translation(self):
        return np.array(self[:3, 3])

    @property
    def rot(self):
        return np.array(self[:3, :3])

    @classmethod
    def identity(cls):
        return cls(np.identity(4))

    @classmethod
    def from_xyz_quat(
        cls,
        x: int | float = 0.0,
        y: int | float = 0.0,
        z: int | float = 0.0,
        q_x: int | float = 0.0,
        q_y: int | float = 0.0,
        q_z: int | float = 0.0,
        w: int | float = 1.0,
    ):
        arr = [x, y, z, q_x, q_y, q_z, w]
        R = RotationSciPy(arr[3:]).as_matrix()
        t = np.array(arr[:3])[:, np.newaxis]
        h = np.array([0, 0, 0, 1])[np.newaxis, :]
        H = np.hstack((R, t))
        H = np.vstack((H, h))
        return cls(H)

    def inverse(self) -> "Pose":
        """
        Calculates the inverse of the Pose using affine information (preferred over np.linalg.inv()).

        :return: inverted Pose
        """
        rot = self.rot
        trans = self.translation

        new_trans = -rot.transpose() @ trans
        return Pose(
            np.vstack(
                [
                    np.hstack(
                        [
                            rot.transpose(),
                            np.array([[new_trans[0]], [new_trans[1]], [new_trans[2]]]),
                        ]
                    ),
                    np.array([0.0, 0.0, 0.0, 1.0]),
                ]
            )
        )