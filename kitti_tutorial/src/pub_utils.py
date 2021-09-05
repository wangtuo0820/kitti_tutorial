from std_msgs.msg import Header
import rospy
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf
import numpy as np
import cv2

from cv_bridge import CvBridge
bridge = CvBridge()

COLOR_DICT = {'Car': (255,255,0), 'Pedestrian': (0, 226, 255), 'Cyclist':(141,40,255)}

LINES = [[0,1],[1,2],[2,3],[3,0]]
LINES+= [[4,5],[5,6],[6,7],[7,4]]
LINES+= [[4,0],[5,1],[6,2],[7,3]]
LINES+= [[4,1],[5,0]]

def pub_img(cam_pub, img, boxes, types):
    for box, typ in zip(boxes, types):
        top_left = int(box[0]), int(box[1])
        bottom_right = int(box[2]), int(box[3])
        cv2.rectangle(img, top_left, bottom_right, COLOR_DICT[typ], 2)
    cam_pub.publish(bridge.cv2_to_imgmsg(img, 'bgr8'))
    rospy.loginfo('camera image published')

def pub_pcl(pcl_pub, point_cloud):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'map'
    pcl_pub.publish(pcl2.create_cloud_xyz32(header, point_cloud[:, :3]))
    rospy.loginfo('lidar point cloud published')

def pub_ego_car(ego_car_pub):
    marker_array = MarkerArray()
    marker = Marker()
    marker.header.frame_id = 'map'
    marker.header.stamp = rospy.Time.now()

    marker.id = 1
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration()
    marker.type = Marker.LINE_STRIP

    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.scale.x = 0.2

    marker.points = []
    marker.points.append(Point(10, -10, 0))
    marker.points.append(Point(0, 0, 0))
    marker.points.append(Point(10, 10, 0))

    marker_array.markers.append(marker)

    mesh_marker = Marker()
    mesh_marker.header.frame_id = 'map'
    mesh_marker.header.stamp = rospy.Time.now()

    mesh_marker.id = 0
    mesh_marker.lifetime = rospy.Duration()
    mesh_marker.type = Marker.MESH_RESOURCE
    mesh_marker.mesh_resource = "file:///home/tuo/catkin_ws/src/kitti_tutorial/truck/Car.dae"

    mesh_marker.pose.position.x = 0.0
    mesh_marker.pose.position.y = 0.0
    mesh_marker.pose.position.z = -1.73 # height of car

    q = tf.transformations.quaternion_from_euler(0, 0, np.pi/2)
    mesh_marker.pose.orientation.x = q[0]
    mesh_marker.pose.orientation.y = q[1]
    mesh_marker.pose.orientation.z = q[2]
    mesh_marker.pose.orientation.w = q[3]

    mesh_marker.color.r = 1.0
    mesh_marker.color.g = 1.0
    mesh_marker.color.b = 1.0
    mesh_marker.color.a = 1.0

    mesh_marker.scale.x = 0.9
    mesh_marker.scale.y = 0.9
    mesh_marker.scale.z = 0.9

    marker_array.markers.append(mesh_marker)

    ego_car_pub.publish(marker_array)
    rospy.loginfo('car model published')

def pub_imu(imu_pub, imu_data):
    imu = Imu()
    imu.header.frame_id = 'map'
    imu.header.stamp = rospy.Time.now()

    q = tf.transformations.quaternion_from_euler(
            float(imu_data.roll),
            float(imu_data.roll),
            float(imu_data.roll))
    imu.orientation.x = q[0]
    imu.orientation.y = q[1]
    imu.orientation.z = q[2]
    imu.orientation.w = q[3]
    imu.linear_acceleration.x = imu_data.af
    imu.linear_acceleration.y = imu_data.al
    imu.linear_acceleration.z = imu_data.au
    imu.angular_velocity.x = imu_data.wf
    imu.angular_velocity.y = imu_data.wl
    imu.angular_velocity.z = imu_data.wu

    imu_pub.publish(imu)
    rospy.loginfo('imu published')

def pub_gps(gps_pub, imu_data):
    gps = NavSatFix()
    gps.header.frame_id = 'map'
    gps.header.stamp = rospy.Time.now()

    gps.latitude = imu_data.lat
    gps.longitude = imu_data.lon
    gps.altitude = imu_data.alt

    gps_pub.publish(gps)
    rospy.loginfo('gps published')

def pub_3dbox(box3d_pub, corners_3d_velos, types, track_ids):
    marker_array = MarkerArray()
    for i, corners_3d_velo in enumerate(corners_3d_velos):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()

        marker.id = i
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration()
        marker.type = Marker.LINE_LIST

        b, g, r = COLOR_DICT[types[i]]
        marker.color.r = r / 255.0
        marker.color.g = g / 255.0
        marker.color.b = b / 255.0 
        marker.color.a = 1.0

        marker.scale.x = 0.1
        marker.points = []
        for l in LINES:
            p1 = corners_3d_velo[l[0]]
            marker.points.append(Point(p1[0],p1[1],p1[2]))
            p2 = corners_3d_velo[l[1]]
            marker.points.append(Point(p2[0],p2[1],p2[2]))
        marker_array.markers.append(marker)


        text_marker = Marker()
        text_marker.header.frame_id = 'map'
        text_marker.header.stamp = rospy.Time.now()

        text_marker.id = i + 1000
        text_marker.action = Marker.ADD
        text_marker.lifetime = rospy.Duration()
        text_marker.type = Marker.TEXT_VIEW_FACING

        p4 = corners_3d_velo[4] # upper front lefr corner
        p = np.mean(corners_3d_velo, axis=0)

        text_marker.pose.position.x = p4[0]
        text_marker.pose.position.y = p4[1]
        text_marker.pose.position.z = p4[2] + 1
        text_marker.text = 'id:' + str(track_ids[i])

        text_marker.scale.x = 1
        text_marker.scale.y = 1
        text_marker.scale.z = 1

        b, g, r = COLOR_DICT[types[i]]
        text_marker.color.r = r / 255.0
        text_marker.color.g = g / 255.0
        text_marker.color.b = b / 255.0 
        text_marker.color.a = 1.0
        marker_array.markers.append(text_marker)

    box3d_pub.publish(marker_array)
    rospy.loginfo('3dbox published')

def pub_loc(loc_pub, tracker, centers):
    marker_array = MarkerArray()

    for track_id in tracker:
        if track_id not in centers and track_id != -1:
            continue
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()

        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration()
        marker.type = Marker.LINE_STRIP
        marker.id = track_id

        marker.color.r = 1.0   
        marker.color.g = 0.0   
        marker.color.b = 0.0    
        marker.color.a = 1.0
        marker.scale.x = 0.2

        marker.points = []
        for p in tracker[track_id].locations:
            marker.points.append(Point(p[0], p[1], 0))

        if len(tracker[track_id].locations) == 1:
            marker.points.append(Point(p[0]+1e-5, p[1]+1e-5, 0))


        marker_array.markers.append(marker)
        loc_pub.publish(marker_array)

def pub_dist(dist_pub, minPQDs):
    marker_array = MarkerArray()

    for i, (minP, minQ, minD) in enumerate(minPQDs):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()

        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration()
        marker.type = Marker.LINE_STRIP
        marker.id = i

        marker.color.r = 1.0   
        marker.color.g = 0.0   
        marker.color.b = 1.0    
        marker.color.a = 0.5
        marker.scale.x = 0.1

        marker.points = []
        marker.points.append(Point(minP[0], minP[1], 0))
        marker.points.append(Point(minQ[0], minQ[1], 0))
        marker_array.markers.append(marker)

        text_marker = Marker()
        text_marker.header.frame_id = 'map'
        text_marker.header.stamp = rospy.Time.now()

        text_marker.id = i + 1000
        text_marker.action = Marker.ADD
        text_marker.lifetime = rospy.Duration()
        text_marker.type = Marker.TEXT_VIEW_FACING

        p = (minP + minQ) / 2.0

        text_marker.pose.position.x = p[0]
        text_marker.pose.position.y = p[1]
        text_marker.pose.position.z = 0
        text_marker.text = 'dist:%.2f'%minD

        text_marker.scale.x = 1
        text_marker.scale.y = 1
        text_marker.scale.z = 1

        text_marker.color.r = 1.0 
        text_marker.color.g = 1.0 
        text_marker.color.b = 1.0  
        text_marker.color.a = 1.0

        marker_array.markers.append(text_marker)
    dist_pub.publish(marker_array)
