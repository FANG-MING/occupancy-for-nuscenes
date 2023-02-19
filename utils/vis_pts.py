import numpy as np

from mayavi import mlab
import argparse
point_cloud_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]
voxel_size=0.2
voxel_shape=(int((point_cloud_range[3]-point_cloud_range[0])/voxel_size),
             int((point_cloud_range[4]-point_cloud_range[1])/voxel_size),
             int((point_cloud_range[5]-point_cloud_range[2])/voxel_size))

def remove_far(points, point_cloud_range):
    mask = (points[:, 0]>point_cloud_range[0]) & (points[:, 0]<point_cloud_range[3]) & (points[:, 1]>point_cloud_range[1]) & (points[:, 1]<point_cloud_range[4]) \
            & (points[:, 2]>point_cloud_range[2]) & (points[:, 2]<point_cloud_range[5])
    return points[mask, :]

def voxelize(voxel: np.array, label_count: np.array):

    for x in range(voxel.shape[0]):
        for y in range(voxel.shape[1]):
            for z in range(voxel.shape[2]):
                if label_count[x, y, z] == 0:
                    continue
                labels = voxel[x, y, z]
                try:
                    label_count[x, y, z] = np.argmax(np.bincount(labels[labels!=0]))
                except:
                    label_count[x, y, z] = 0
    return label_count

def points2voxel(points, voxel_shape, voxel_size, max_points=5, specific_category=None):
    voxel = np.zeros((*voxel_shape, max_points), dtype=np.int64)
    label_count = np.zeros((voxel_shape), dtype=np.int64)
    index = points[:, 4].argsort()
    points = points[index]
    for point in points:
      
        x, y, z = point[0], point[1], point[2]
        x = round((x - point_cloud_range[0]) / voxel_size)
        y = round((y - point_cloud_range[1]) / voxel_size)
        z = round((z - point_cloud_range[2]) / voxel_size)

        try:
            voxel[x, y, z, label_count[x, y, z]] = int(point[4])  # map_label[int(point[4])]
            label_count[x, y, z] += 1
        except:
            continue

    voxel = voxelize(voxel, label_count)
    voxel = voxel.astype(np.float64)
    return voxel


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid


def draw(
    voxels,
    voxel_size=0.2,
):


    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T


    grid_voxels = grid_coords[
        (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)
    ]

    figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))

    
    plt_plot = mlab.points3d(
        grid_voxels[:, 0],
        grid_voxels[:, 1],
        grid_voxels[:, 2],
        grid_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    classname_to_color = {  # RGB.
        "noise": (0, 0, 0),  # Black.
        "animal": (70, 130, 180),  # Steelblue
        "human.pedestrian.adult": (0, 0, 230),  # Blue
        "human.pedestrian.child":(0, 0, 230),  # Skyblue,
        "human.pedestrian.construction_worker":(0, 0, 230),  # Cornflowerblue
        "human.pedestrian.personal_mobility": (0, 0, 230),  # Palevioletred
        "human.pedestrian.police_officer":(0, 0, 230),  # Navy,
        "human.pedestrian.stroller": (0, 0, 230),  # Lightcoral
        "human.pedestrian.wheelchair": (0, 0, 230),  # Blueviolet
        "movable_object.barrier": (112, 128, 144),  # Slategrey
        "movable_object.debris": (112, 128, 144),  # Chocolate
        "movable_object.pushable_pullable":(112, 128, 144),  # Dimgrey
        "movable_object.trafficcone":(112, 128, 144),  # Darkslategrey
        "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
        "vehicle.bicycle": (220, 20, 60),  # Crimson
        "vehicle.bus.bendy":(255, 158, 0),  # Coral
        "vehicle.bus.rigid": (255, 158, 0),  # Orangered
        "vehicle.car": (255, 158, 0),  # Orange
        "vehicle.construction":(255, 158, 0),  # Darksalmon
        "vehicle.emergency.ambulance":(255, 158, 0),
        "vehicle.emergency.police": (255, 158, 0),  # Gold
        "vehicle.motorcycle": (255, 158, 0),  # Red
        "vehicle.trailer":(255, 158, 0),  # Darkorange
        "vehicle.truck": (255, 158, 0),  # Tomato
        "flat.driveable_surface": (0, 207, 191),  # nuTonomy green
        "flat.other":(0, 207, 191),
        "flat.sidewalk": (75, 0, 75),
        "flat.terrain": (0, 207, 191),
        "static.manmade": (222, 184, 135),  # Burlywood
        "static.other": (0, 207, 191),  # Bisque
        "static.vegetation": (0, 175, 0),  # Green
        "vehicle.ego": (255, 240, 245)
    }
    colors = np.array(list(classname_to_color.values())).astype(np.uint8)
    alpha = np.ones((colors.shape[0], 1), dtype=np.uint8) * 255
    colors = np.hstack([colors, alpha])



    plt_plot.glyph.scale_mode = "scale_by_vector"

    plt_plot.module_manager.scalar_lut_manager.lut.table = colors
    plt_plot.module_manager.scalar_lut_manager.data_range = [0, 31]
    mlab.show()



# points = remove_far(points, point_cloud_range)
def main(path):
    points = np.fromfile(path, dtype=np.float16).reshape(-1, 5)

    y_pred = points2voxel(points, voxel_shape, voxel_size, 20)
    draw(
        y_pred[::-1, ::-1, :],
        voxel_size=0.2,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pts-path', required=True)
    parser.add_argument('--voxel-size', type=float, default=0.2)

    args = parser.parse_args()
    voxel_size = args.voxel_size
    voxel_shape=(int((point_cloud_range[3]-point_cloud_range[0])/voxel_size),
             int((point_cloud_range[4]-point_cloud_range[1])/voxel_size),
             int((point_cloud_range[5]-point_cloud_range[2])/voxel_size))
    main(args.pts_path)