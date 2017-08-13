#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    
    # TODO: Statistical Outlier Filtering
    outlier_filter = cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(5)   # no. of neighbouring points
    x = 0.1     #threshold scale factor
    outlier_filter.set_std_dev_mul_thresh(x) # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    cloud_filtered = outlier_filter.filter()

    # TODO: Voxel Grid Downsampling
    # Create the filter, set downsample size to 1cm and apply filter
    vox = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # TODO: PassThrough Filter
    # Create the filter along Z direction for table & objects (0.6-1m) and apply filter
    pass_thru = cloud_filtered.make_passthrough_filter()
    axis_min = 0.6
    axis_max = 1.0
    pass_thru.set_filter_field_name('z')
    pass_thru.set_filter_limits(axis_min, axis_max)
    cloud_filtered = pass_thru.filter()

    # Create the filter along X (in front of robot) to filter out box edges (0.4-1m) and apply filter
    pass_thru = cloud_filtered.make_passthrough_filter()
    axis_min = 0.4
    axis_max = 1.0
    pass_thru.set_filter_field_name('x')
    pass_thru.set_filter_limits(axis_min, axis_max)
    cloud_filtered = pass_thru.filter()

    # TODO: RANSAC Plane Segmentation
    # Fitting a plane to include the table top points
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    # Separate/extract the objects and the table top
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)

    # TODO: Euclidean Clustering
    # Remove color info to create a KD-tree and extract cluster of object points
    # using cluster size (10-5000) and 5cm cluster tolerance
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(5000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])])
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_cluster_pub.publish(ros_cluster_cloud)

# Exercise-3 TODOs:
    # Lists to hold the detected labels and objects
    detected_objects_labels = []
    detected_objects_list = []

    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        # Compute the hsv and normals histogram and concatenate them
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        # Scale the feature and predict using the scaler, classfier and encoder respectively
        # created during training to get the label and append it to the labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        # Set the position of the label slightly above (40cm) the object
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects_list.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects_list)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects_list)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    output_dict_list = []
    # Create a dictionary of detected objects for easy lookup of point cloud using label
    detected_obj_dict = {}
    for do in object_list:
        detected_obj_dict[do.label] = do.cloud

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    # TODO: Parse parameters into individual variables
    # Create a dictionary of dropbox for easy lookup of position using group as key
    dropbox_dict = {}
    for box in dropbox_param:
        dropbox_dict[box['group']] = box['position']

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    for pl_obj in object_list_param:
        # Pick list object values
        pl_obj_name = pl_obj['name']
        pl_obj_group = pl_obj['group']

        # Skip if the object is not in the list of detected objects
        if pl_obj_name not in detected_obj_dict:
            continue

        # Set the ROS test scene and object name parameters
        test_scene_num = Int32()
        test_scene_num.data = 1
        object_name = String()
        object_name.data = pl_obj_name

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        cloud = detected_obj_dict[pl_obj_name]
        points_arr = ros_to_pcl(cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[:3]

        # Create 'pick_pose' and set position by recasting centroid to python float
        pick_pose = Pose()
        pick_pose.position.x = np.asscalar(centroid[0])
        pick_pose.position.y = np.asscalar(centroid[1])
        pick_pose.position.z = np.asscalar(centroid[2])

        # TODO: Create 'place_pose' for the object
        # Set it's position to the corresponding dropbox group position
        place_pose = Pose()
        place_pose.position.x = dropbox_dict[pl_obj_group][0]
        place_pose.position.y = dropbox_dict[pl_obj_group][1]
        place_pose.position.z = dropbox_dict[pl_obj_group][2]

        # TODO: Assign the arm to be used for pick_place
        arm_name = String()
        if pl_obj_group == 'green':
            arm_name.data = 'right'
        else:
            arm_name.data = 'left'

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        output_dict_list.append(yaml_dict)

        """
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)
            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
        """

    # TODO: Output your request parameters into output yaml file
    send_to_yaml('../config/output_' + str(test_scene_num.data) +'.yaml', output_dict_list)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('pick_place', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher('/pcl_objects', PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher('/pcl_cluster', PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher('/object_markers', Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher('/detected_objects', DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
