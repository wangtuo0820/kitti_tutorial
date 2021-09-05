import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
from kitti_utils import *

mpl.rcParams['legend.fontsize'] = 10

TRACKING_COLUMN_NAMES = [
        'frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha',
        'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
        'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']


frame = 0
DATA_PATH = '/media/tuo/笔记本硬盘/kitti/demo/2011_09_26/2011_09_26_drive_0005_sync'
points = np.fromfile(os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin'%(frame%154)), dtype=np.float32).reshape(-1, 4)

def load_tracking(path):
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = TRACKING_COLUMN_NAMES
    df.loc[df.type.isin(['Truck', 'Van', 'Tram']), 'type'] = 'Car'
    df = df[df.type.isin(['Car', 'Pedestrain', 'Cyclist'])]
    return df


def compute_3d_box_cam2(h,w,l,x,y,z,yaw):
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0,1,0],[-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l/2, l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [0,   0,    0,    0,   -h,   -h,   -h,   -h  ]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2  ]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d_cam2 += np.vstack([x,y,z])
    return corners_3d_cam2 
    


def draw_point_cloud(ax, title, axes=[0, 1, 2], point_size=0.1, xlim3d=None, ylim3d=None, zlim3d=None):
        """
        Convenient method for drawing various point cloud projections as a part of frame statistics.
        """
        
        axes_limits = [
            [-20, 80], # X axis range
            [-20, 20], # Y axis range
            [-3, 10]   # Z axis range
        ]
        axes_str = ['X', 'Y', 'Z']
        ax.grid(False)

        ax.scatter(*np.transpose(points[:, axes]), s=point_size, c=points[:, 3], cmap='gray')
        ax.set_title(title)
        ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
        ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
        if len(axes) > 2:
            ax.set_xlim3d(*axes_limits[axes[0]])
            ax.set_ylim3d(*axes_limits[axes[1]])
            ax.set_zlim3d(*axes_limits[axes[2]])
            ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
        else:
            ax.set_xlim(*axes_limits[axes[0]])
            ax.set_ylim(*axes_limits[axes[1]])
        # User specified limits
        if xlim3d!=None:
            ax.set_xlim3d(xlim3d)
        if ylim3d!=None:
            ax.set_ylim3d(ylim3d)
        if zlim3d!=None:
            ax.set_zlim3d(zlim3d)
            
def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.
    
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)

 
if __name__ == '__main__': 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(40, 150)
    draw_point_cloud(ax, points[::5])

    df_tracking = load_tracking('/media/tuo/笔记本硬盘/kitti/data_tracking_label_2/training/label_02/0000.txt')
    corners_3d_cam2 = compute_3d_box_cam2(*df_tracking.loc[2, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']]) # (3,8)

    calib = Calibration('/media/tuo/笔记本硬盘/kitti/demo/2011_09_26/', from_video=True)
    corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T).T
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax.view_init(40, 150)
    draw_box(ax, corners_3d_velo)

    
    #fig, ax = plt.subplots()
    #draw_point_cloud(ax, points[::5], axes=[0,1])
     
    plt.show()
