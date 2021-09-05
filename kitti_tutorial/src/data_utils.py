import numpy as np
import cv2, os
import pandas as pd


DATA_PATH = '/media/tuo/笔记本硬盘/kitti/demo/2011_09_26/2011_09_26_drive_0005_sync'

IMU_COLUMN_NAMES = [
        'lat', 'lon', 'alt', 'roll', 'pitch', 'yaw',
        'vn', 've', 'vf', 'vl', 'vu',  
        'ax', 'ay', 'az', 'af', 'al', 'au', 
        'wx', 'wy', 'wz', 'wf', 'wl', 'wu',  
        'posacc', 'velacc', 
        'navstat', 'numsats', 
        'posmode', 'velmode', 'orimode']

TRACKING_COLUMN_NAMES = [
        'frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha',
        'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
        'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']


def load_img(frame):
    return cv2.imread(os.path.join(DATA_PATH, 'image_02/data/%010d.png' %(frame % 154) ))

def load_pcl(frame):
    return np.fromfile(os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin'%(frame%154)), dtype=np.float32).reshape(-1, 4)

def load_imu(frame):
    path = os.path.join(DATA_PATH, 'oxts/data/%010d.txt' % (frame % 154))
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = IMU_COLUMN_NAMES
    return df

def load_tracking(path):
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = TRACKING_COLUMN_NAMES
    df.loc[df.type.isin(['Truck', 'Van', 'Tram']), 'type'] = 'Car'
    df = df[df.type.isin(['Car', 'Pedestrain', 'Cyclist'])]
    return df
