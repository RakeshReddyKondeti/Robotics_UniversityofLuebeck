#!/usr/bin/env python
"""
@author: Lars Schilling

"""
#imports
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_data():
	#get map and metadata to transfrom into a matrix

	msg=rospy.wait_for_message('/map_image/tile', Image)
	odom=rospy.wait_for_message('/odometry/gps', Odometry)
	#odom=rospy.wait_for_message('/pose_gt', Odometry)
	metadata=rospy.wait_for_message('/map_metadata', MapMetaData)
	resolution=metadata.resolution

	bridge = CvBridge()
	data = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
	data=np.array(data)
	data[data==0] = 1
	data[data==127] = 2
	data[data==255] = 0

	return data, odom, resolution

def frontier_exploration():

	data, odom, resolution = get_data()
	img_work = data	
	
	#visualize data
	plt.subplot(221)	
	plt.imshow(img_work)
	plt.title('Original Image')

	#find all frontier points, can be defined as edge detection problem, cv2.Canny can be used
	edges = cv2.Canny(img_work, 11, 11, L2gradient = False)	
	plt.subplot(222)
	plt.imshow(edges)
	plt.title('Edged Image')

	#calculate information gain, tip: set everything to zero expect unexplored area
	#you can use cv2.filter2D with your own kernel
	info_gain = np.zeros((64,64), dtype=np.float64)
	indices = np.where(img_work == 2)
	info_gain[indices] = 2
	
	kernel = np.ones((9,9),np.float64)/25
	conv_info_gain = cv2.filter2D(info_gain,-1,kernel)
	
	plt.subplot(223)
	plt.imshow(conv_info_gain)
	plt.title('Convoluted on unexplored area')

	#find the frontier point with the biggest information gain, this will be our goal point
	max_info_indices = np.where(conv_info_gain == np.max(conv_info_gain))	
	listOfCoordinates = list(zip(max_info_indices[1],max_info_indices[0]))

	max_info_gain_image = cv2.circle(edges, listOfCoordinates[len(listOfCoordinates)-1], radius = 0, color=(139,0,0), thickness=-1)
	count = 0
	for index in listOfCoordinates:
		print('Maximum information gain is found at coordinate: ' + str(index))
		count = count + 1
	
	print('\n')
	print('Found ' + str(count) + ' coordinates with maximum information gain of ' + str(np.max(conv_info_gain)))
	print('Marking maximum information gain on point: ' + str(listOfCoordinates[len(listOfCoordinates) - 1]))

	plt.subplot(224)
	plt.imshow(max_info_gain_image)
	plt.title('Highest info gain at ' + str(listOfCoordinates[len(listOfCoordinates) - 1]))	
	
	plt.show()
	

	#define a PoseStamped message here and publish it on the move_base_publisher
	goal=PoseStamped()
	goal.header.stamp=rospy.Time.now()
	goal.header.frame_id="odom"
	goal.pose.orientation.w=1
	#define x and y position of the pose here
	goal.pose.position.x = listOfCoordinates[len(listOfCoordinates)-1][0]
	goal.pose.position.y = listOfCoordinates[len(listOfCoordinates)-1][1]
	#use the odom position and the goal point
	goal.pose.position.x = goal.pose.position.x - odom[0]
	goal.pose.position.y = goal.pose.position.y - odom[1]
	#reminder: the "odom position of the boat" is the center of the image!
	#the goal point must still be converted, you will need the resolution of the map for this which is given with the parameter "resolution"
	goal.pose.position.x = goal.pose.position.x/resolution
	goal.pose.position.y = goal.pose.position.y/resolution

	move_base_publisher.publish(goal)

if __name__ == '__main__':
	try:

		rospy.init_node('frontier_exploration')
		move_base_publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
		while not rospy.is_shutdown():
			frontier_exploration()
	except rospy.ROSInterruptException: pass
