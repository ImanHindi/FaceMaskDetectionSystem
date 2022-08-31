# USAGE
#python ImageShotMaskDetection.py --image C:\Users\user\Desktop\iman\FaceMaskDetection-SocialDistancing\Dataset\FMD_DATASET\demo-images
#python ImageShotMaskDetection.py --image C:\Users\user\Downloads\Medicalmask\Medicalmask\MedicalMask
#python ImageShotMaskDetection.py --image C:\Users\user\Desktop\iman\Face-Mask-Detection\images\
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
		# load the input image from disk, clone it, and send it to the model API
		#  for preprocessing and detection
		print("[INFO] classifying {}".format(
		imagePath[imagePath.rfind("/") + 1:]))
		image = cv2.imread(imagePath)
		cv2.imwrite('image.png',image)
		(locs, preds,preds_actual) = prediction_request('image.png',source=1)
		
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
						label = "Mask"
						color = (0, 255, 0)
				elif np.argmax(pred)==2:
						label = "withoutMask"
						color = (0, 0, 255)
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
				# Save the output image
		cv2.imwrite(f"image_MaskDetection'{label}''{i}'.png",image)
		i=i+1
		#cv2.imshow("Output", image)
		cv2.waitKey(0)

if __name__ == "__main__":
	mask_image()
prepare_and_save_report(source=1)