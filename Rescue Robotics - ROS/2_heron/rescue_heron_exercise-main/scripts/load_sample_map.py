#!/usr/bin/env python
"""
@author: Lars Schilling

"""
import rospkg
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def transform_data(data):

	data=np.array(data)
	data[data==0] = 1
	data[data==127] = 2
	data[data==255] = 0
	

	return data



if __name__ == '__main__':
	rospack = rospkg.RosPack()
	img = Image.open(rospack.get_path('heron_exploration')+'/sample_maps/sample_map_5.png')
	img_work = transform_data(img)

	plt.subplot(221)	
	plt.imshow(img_work)
	plt.title('Original Image')

	###try frontier point detection with img_work
	#find all frontier points, can be defined as edge detection problem, cv2.Canny can be used
	
	edges = cv2.Canny(img_work, 11, 11, L2gradient = False)
	#print(np.max(edges))	
	#plt.imshow(edges)
	#plt.show()

	plt.subplot(222)
	plt.imshow(edges)
	plt.title('Edged Image')

	#print("2D array of img_work: ",img_work)
	#print("2D array of edged image: ",edges)

	#calculate information gain, tip: set everything to zero expect unexplored area	

	#copying all the unexplored area into info_gain and setting everything to zero
	info_gain = np.zeros((64,64), dtype=np.float64)
	indices = np.where(img_work == 2)
	#print(indices)
	
	info_gain[indices] = 2
 
	[x,y] = np.shape(img_work)

	#plt.imshow(info_gain)
	#plt.show()

	#you can use cv2.filter2D with your own kernel
	kernel = np.ones((9,9),np.float64)/25
	conv_info_gain = cv2.filter2D(info_gain,-1,kernel)
	#blur = cv2.blur(dst,(5,5))
	#plt.imshow(conv_info_gain)
	#plt.show()
	
	plt.subplot(223)
	plt.imshow(conv_info_gain)
	plt.title('Convoluted on unexplored area')
	#find the frontier point with the biggest information gain, this will be our goal point
	
	#travering all the convolved info_gain 2D array to find the maximum coordinates and corresponding indices
	max_info_indices = np.where(conv_info_gain == np.max(conv_info_gain))	
	listOfCoordinates = list(zip(max_info_indices[1],max_info_indices[0]))
	#print(listOfCoordinates[0])
	#print('Maximum information gain is: ',np.max(conv_info_gain))
				
	
	#marking a max_info_point on the edged image
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
	
	
	



