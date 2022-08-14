# USAGE
# python detect_mask_image.py --image C:\Users\user\Desktop\iman\FinalProject\Dataset\FMD_DATASET\test\1

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
from Client.MaskDetectionRequest import prediction_request

def mask_image():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")

	args = vars(ap.parse_args())

	
	# load the input image from disk, clone it, and grab the image spatial
	# dimensions
	image = cv2.imread(args["image"])
	orig = image.copy()



	(locs, preds) = prediction_request(image)
	
# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask,incorrect_Mask) = pred
		if mask == 1:
			label = "Mask"
			color = (0, 255, 0)
		elif withoutMask == 1:
			label = "NoMask"
			color = (0, 0, 255)
		else:
			label = "incorrectMask"
			color = (255, 0, 0)
			
		# include the probability in the label
		label= "{}: {:.2f}%".format(label, max(mask, withoutMask,incorrect_Mask) * 100)

			
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
