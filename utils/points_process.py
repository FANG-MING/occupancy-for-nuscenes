import numpy as np
from functools import partial
from nuscenes.utils.geometry_utils import points_in_box
def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def translate(points, x: np.ndarray) -> np.array:
    """
    Applies a translation to the point cloud.
    :param x: <np.float: 3, 1>. Translation in x, y, z.
    """
    for i in range(3):
        points[i, :] = points[i, :] + x[i]
    return points


def rotate(points, rot_matrix: np.ndarray, center=None) -> np.array:
    """
    Applies a rotation.
    :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
    """
    if center is not None:
        points[:3, :] = np.dot(rot_matrix, points[:3, :]-center[:, None]) + center[:, None]
    else:

        points[:3, :] = np.dot(rot_matrix, points[:3, :])
    return points


def transform(points, rotate_matrix: np.ndarray, translation_matrix, inverse=False) -> None:
    """
    Applies a homogeneous transform.
    :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
    """
    if not inverse:
        points = rotate(points, rotate_matrix)
        points = translate(points, translation_matrix)
    else:
        points = translate(points, -translation_matrix)
        points = rotate(points, np.linalg.inv(rotate_matrix))
    return points

def remove_close(points, radius: tuple=(1.0, 1.5)):
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """

    x_filt = np.abs(points[0, :]) < radius[0]
    y_filt = np.abs(points[1, :]) < radius[1]
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points
