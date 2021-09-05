#!/usr/bin/env python 

import rospy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pcl2
import cv2, os
import numpy as np
from collections import deque

from data_utils import *
from pub_utils import *
from kitti_utils import *

EGO_CAR = np.array([[2.15,0.9,-1.73], [2.15,-0.9,-1.73], [-1.95, -0.9, -1.73], [-1.95, 0.9, -1.73],
                    [2.15,0.9,-0.23], [2.15,-0.9,-0.23], [-1.95, -0.9, -0.23], [-1.95, 0.9, -0.23]])

def compute_3d_box_cam2(h,w,l,x,y,z,yaw):
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0,1,0],[-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l/2, l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [0,   0,    0,    0,   -h,   -h,   -h,   -h  ]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2  ]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d_cam2 += np.vstack([x,y,z])
    return corners_3d_cam2 

def compute_great_circle_distance(lat1, lon1, lat2, lon2):
    delta_sigma = float(np.sin(lat1*np.pi/180)*np.sin(lat2*np.pi/180) + 
                        np.cos(lat1*np.pi/180)*np.cos(lat2*np.pi/180)*np.cos(lon1*np.pi/180 - lon2*np.pi/180))
    return 6371000.0 * np.arccos(np.clip(delta_sigma, -1, 1))

def distance_point_to_segment(P, A, B):
    AP = P - A
    BP = P - B
    AB = B - A
    if np.dot(AB, AP) >= 0 and np.dot(-AB, BP) >= 0: # project point in rectangle
        return np.abs(np.cross(AP, AB)) / np.linalg.norm(AB), np.dot(AP, AB) / np.dot(AB, AB) * AB + A
    d_PA = np.linalg.norm(AP)
    d_PB = np.linalg.norm(BP)
    if d_PA < d_PB:
        return d_PA, A
    return d_PB, B

def min_distance_cuboids(cub1, cub2):
    minD = 1e5
    for i in range(4):
        for j in range(4):
            d, Q = distance_point_to_segment(cub1[i,:2], cub2[j,:2], cub2[j+1,:2])
            if d < minD:
                minD = d
                minP = cub1[i,:2]
                minQ = Q
    for i in range(4):
        for j in range(4):
            d, Q = distance_point_to_segment(cub2[i,:2], cub1[j,:2], cub1[j+1,:2])
            if d < minD:
                minD = d
                minP = cub2[i,:2]
                minQ = Q
    return minP, minQ, minD

class Object():
    def __init__(self, center):
        self.locations = deque(maxlen=20)
        self.locations.appendleft(center)

    def update(self, center, displacement, yaw):
        for i in range(len(self.locations)):
            x0, y0 = self.locations[i]
            x1 = x0 * np.cos(yaw_change) + y0 * np.sin(yaw_change) - displacement
            y1 = -x0 * np.sin(yaw_change) + y0 * np.cos(yaw_change)
            self.locations[i] = np.array([x1, y1])

        if center is not None:
            self.locations += [center]
            # self.locations.appendleft(center)

    def reset(self):
        self.locations = deque(maxlen=20)
    

if __name__ == '__main__':
    rospy.init_node('kitti_node', anonymous=True)
    cam_pub = rospy.Publisher('kitti_cam', Image, queue_size=10)
    pcl_pub = rospy.Publisher('kitti_point_cloud', PointCloud2, queue_size=10)
    ego_car_pub = rospy.Publisher('kitti_ego_car', MarkerArray, queue_size=10)
    imu_pub = rospy.Publisher('kitti_imu', Imu, queue_size=10)
    gps_pub = rospy.Publisher('kitti_gps', NavSatFix, queue_size=10) 
    box3d_pub =  rospy.Publisher('kitti_box3d', MarkerArray, queue_size=10)
    loc_pub = rospy.Publisher('kitti_loc', MarkerArray, queue_size=10)
    dist_pub = rospy.Publisher('kitti_dist', MarkerArray, queue_size=10)

    df_tracking = load_tracking('/media/tuo/笔记本硬盘/kitti/data_tracking_label_2/training/label_02/0000.txt')
    calib = Calibration('/media/tuo/笔记本硬盘/kitti/demo/2011_09_26/', from_video=True)

    tracker = {} # track_id: Object
    prev_imu_data = None

    rate = rospy.Rate(10)
    frame = 0
    while not rospy.is_shutdown():
        frame = frame % 154
        df_tracking_frame = df_tracking[df_tracking.frame == frame]

        boxes = np.array(df_tracking_frame[['bbox_left','bbox_top','bbox_right','bbox_bottom']])
        types = np.array(df_tracking_frame['type'])
        boxes_3d = np.array(df_tracking_frame[['height','width','length','pos_x', 'pos_y', 'pos_z', 'rot_y']])
        track_ids = np.array(df_tracking_frame['track_id']) 

        corners_3d_velos = []
        centers = {} # track_id: center
        minPQDs = []

        for track_id, box_3d in zip(track_ids, boxes_3d):
            corners_3d_cam2  = compute_3d_box_cam2(*box_3d)
            corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
            corners_3d_velos += [corners_3d_velo]
            centers[track_id] = np.mean(corners_3d_velo, axis=0)[:2]
            minPQDs += [min_distance_cuboids(EGO_CAR, corners_3d_velo)]
        # add ego car
        corners_3d_velos += [EGO_CAR]
        types = np.append(types, 'Car')
        track_ids = np.append(track_ids, -1)
        centers[-1] = np.array([0,0])

        img = load_img(frame)
        point_cloud = load_pcl(frame)
        imu_data = load_imu(frame)

        if prev_imu_data is None:
            for track_id in centers:
                tracker[track_id] = Object(centers[track_id])
        else:
            # displacement = 0.1*np.linalg.norm(imu_data[['vf', 'vl']])
            displacement = compute_great_circle_distance(imu_data.lat, imu_data.lon, prev_imu_data.lat, prev_imu_data.lon)
            yaw_change = float(imu_data.yaw - prev_imu_data.yaw)
            for track_id in centers:
                if track_id in tracker:
                    tracker[track_id].update(centers[track_id], displacement, yaw_change)
                else:
                    tracker[track_id] = Object(centers[track_id])
            for track_id in tracker:
                if track_id not in centers:
                    tracker[track_id].update(None, displacement, yaw_change)
        prev_imu_data = imu_data

        pub_img(cam_pub, img, boxes, types)
        pub_pcl(pcl_pub, point_cloud)
        pub_3dbox(box3d_pub, corners_3d_velos, types, track_ids)
        pub_ego_car(ego_car_pub)
        pub_imu(imu_pub, imu_data)
        pub_gps(gps_pub, imu_data)
        pub_loc(loc_pub, tracker, centers)
        pub_dist(dist_pub, minPQDs)


        rate.sleep()
        frame += 1
        if frame == 154:
            for track_id in tracker:
                tracker[track_id].reset()
                prev_imu_data = None
