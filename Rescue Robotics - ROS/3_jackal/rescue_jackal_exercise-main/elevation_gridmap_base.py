# -*- coding: utf-8 -*-
"""
@author: Lars Schilling

"""
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt
import math

#add imports for the pathfinding package
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

def callback(msg):
   	print('Recieved map data')
    bridge = CvBridge()
    data = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
	height_map=np.array(data, dtype=np.uint)
	[x,y] = np.shape(height_map)
	
		
    #add a nonlinearity to the height map to create a height cost map (low points very low cost, high points very high cost)
	#increasing all values of height map by 1 => all the zeros are converted to 1
	height_map = height_map + 1
	height_map[np.where(height_map == 0)] = 1
	
	#adding nonlinearity to height map by performing element-wise squaring to produce heigth cost map
	height_map = np.power(height_map, 2)
	
	
	print ("---------------------printing height cost map----------------")
	plt.subplot(131)
	plt.imshow(height_map)
	plt.title("Height cost map")
	print("--------------------------------------------------------------")


    #create a distance cost map with the shape of height_costmap
    #pixels around the center should have low cost and get higher at the edges
	#initialising distance map
	distance_map = np.zeros((x,y), dtype=np.uint)

	#filling the distance map with the distances from the center to the pixel cooridinates to produce distance cost map 
	for i in range(x):
		for j in range(y):
			point1 = np.array((x/2 - 0.5, y/2 - 0.5), dtype=np.float64)
			point2 = np.array((i,j), dtype=np.float64)
			distance = np.linalg.norm(point1-point2)
			distance_map[i][j] = distance 

	
	maximum = np.max(distance_map)
	minimum = np.min(distance_map)
	ranges = maximum - minimum
	
	#normalising the distacne cost map to make it scalable(comparable) with the height cost map 
	
	for i in range(x):
		for j in range(y):
			distance_map[i][j] = 255* (distance_map[i][j] - minimum)/ranges
	
	distance_map = np.power(distance_map,2)
	

	print ("---------------------printing distance map----------------")
	print(distance_map)
	plt.subplot(132)
	plt.imshow(distance_map)
	plt.title("Distance map")
	print("--------------------------------------------------------------")
	

    #define a combined cost map based height and distance cost map
    #this could be a sum or multiplication
	
	combined_map = height_map + distance_map
	print ("---------------------printing combined map----------------")
	print(combined_map)
	print("--------------------------------------------------------------")


    #implement the AStarFinder from the pathfinding package to the combined cost map
    #to find a path from the center of the image (30,30) to the upper edge of the image (30,0)

	grid = Grid(matrix= combined_map)
	start = grid.node(30,30)
	end = grid.node(30,0)
	finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
	path, runs = finder.find_path(start, end, grid)
	print('operations:', runs, 'path length:', len(path))
	print(grid.grid_str(path=path, start=start, end=end))
	print(path)
	path = np.array(path, dtype=np.uint8)
	
	plt.subplot(133)
	plt.imshow(combined_map)
	for index in range(len(path)):	
		plt.scatter(path[index][0], path[index][1], marker =',', c='red', linewidth = 0.01)
	
	plt.title("Astar Path")	
	plt.show()
	

    #plot your height, distance and combined cost map, as well as the astar path
    #plt.imshow(height_map)
    #plt.show()
    rospy.sleep(0.5)

if __name__ == '__main__':
    try:

        rospy.init_node('elevation_path')
        sub=rospy.Subscriber("/grid_map_image", Image, callback, queue_size=1)
        rospy.spin()
    except KeyboardInterrupt or rospy.ROSInterruptException: pass
