# USAGE
# python ImageShotMaskDetection.py --image C:\Users\user\Desktop\iman\English\lightness1.jpg
# import the necessary packages
import json
from urllib import response
import numpy as np
import argparse
import cv2
import os
from MaskDetectionRequest import prediction_request

def mask_image():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")

	args = vars(ap.parse_args())

	
	# load the input image from disk, clone it, and grab the image spatial
	# dimensions
	image = cv2.imread(args["image"])
	#orig = image.copy()

	#cv2.imshow("image",image)
	#cv2.waitKey()
	#cv2.destroyAllWindows()
	cv2.imwrite('image.png',image)
	(locs,preds) = prediction_request('image.png')
	#print(locs)
	#print(preds)
	#print(type(response))
	#(list(json.dumps(response['locs'])),list(json.dumps(response['preds'])))
# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		
		print(pred)
		#(incorrect_Mask,mask, withoutMask) = pred
		if np.argmax(pred)==0: 
			label = "incorrectMask"
			color = (255, 0, 0)
		elif np.argmax(pred)==1:
			label = "Mask"
			color = (0, 255, 0)
		elif np.argmax(pred)==2:
			label = "withoutMask"
			color = (0, 0, 255)
			
		# include the probability in the label
		label= "{}: {:.2f}%".format(label, max(pred) * 100)

			
		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

	# show the output image
	cv2.imshow("Output", image)
	cv2.waitKey(0)
	
if __name__ == "__main__":
	mask_image()
