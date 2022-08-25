# USAGE
# python ImageShotMaskDetection.py --images C:\Users\user\Downloads\images
# C:\Users\user\Desktop\iman\Face-Mask-Detection\images\
# import the necessary packages
import json
from urllib import response
import numpy as np
import argparse
import cv2
import os
from MaskDetectionRequest import prediction_request
from SaveReport import prepare_and_save_report

from imutils import paths
def mask_image():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--images", required=True,
		help="path to input images")

	args = vars(ap.parse_args())
	i=0
	for imagePath in paths.list_images(args["images"]):
		# load the input image from disk, clone it, and grab the image spatial
		# dimensions
		print("[INFO] classifying {}".format(
		imagePath[imagePath.rfind("/") + 1:]))
		image = cv2.imread(imagePath)
		#orig = image.copy()

		#cv2.imshow("image",image)
		#cv2.waitKey()
		#cv2.destroyAllWindows()
		cv2.imwrite('image.png',image)
		(locs, preds,preds_actual) = prediction_request('image.png',source=1)
		#print(locs)
		#print(preds)
		#print(type(response))
		#(list(json.dumps(response['locs'])),list(json.dumps(response['preds'])))
		# loop over the detected face locations and their corresponding
		# locations
		if (locs, preds):
			for (box, pred,pred_actual) in zip(locs, preds,preds_actual):
				# unpack the bounding box and predictions
				(startX, startY, endX, endY) = box

				print(pred)
				#(incorrect_Mask,mask, withoutMask) = pred
				if np.argmax(pred)==0: 
					label = "incorrectMask"
					color = (255, 0, 0)
				elif np.argmax(pred)==1:
					if pred[1]>.65:
						label = "Mask"
						color = (0, 255, 0)
					else:
						label="incorrectMask"
						color = (255, 0, 0)
					
				elif np.argmax(pred)==2:
					if pred[2]>.65:
						label = "withoutMask"
						color = (0, 0, 255)
					else:
						label="incorrectMask"
						color = (255, 0, 0)
	
					
					
				else: 
					label = ""
					color = (0, 0, 0)

				# include the probability in the label
				label= "{}: {:.2f}%".format(label, max(pred) * 100)
				face = image[startY:endY, startX:endX]

				# display the label and bounding box rectangle on the output
				# frame
				cv2.imwrite(f"face'{i}'.png",face)
				cv2.putText(image, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
				

				# show the output image
		
		cv2.imwrite(f"image_MaskDetection'{i}'.png",image)
		i=i+1
		#cv2.imshow("Output", image)
		cv2.waitKey(0)

if __name__ == "__main__":
	mask_image()
prepare_and_save_report(source=1)