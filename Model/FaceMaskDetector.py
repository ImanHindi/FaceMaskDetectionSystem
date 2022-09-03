from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import image_utils
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import mediapipe as mp
from mediapipe.modules.face_detection import face_detection_pb2
from mediapipe.calculators.image import image_cropping_calculator_pb2
from mediapipe.framework.formats.detection_pb2 import Detection

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-f", "--face", type=str,
#	default="C:\\Users\\user\\Desktop\\iman\\FinalProject\\Resources\\face_detector",
#	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="VGG19_FaceMaskDetector.hdf",
	help="path to trained face mask detector model")
args = vars(ap.parse_args())
class FaceMaskDetector():
	#load mediapipe face detector
	print("[INFO] loading face detector model...")
	faceNet = mp.solutions.face_detection
	mp_drawing = mp.solutions.drawing_utils
	
	
	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	maskNet = load_model('C:\\Users\\user\\Desktop\\iman\\FaceMaskDetection-SocialDistancing\\Model\\VGG19_FaceMaskDetector.hdf')
	
	def detect_and_predict_mask(frame, faceNet=faceNet, maskNet=maskNet):
		faces = []
		locs = []
		preds = [] 
		preds_actual=[] 
		#face detection using mediapipe detector model
		with faceNet.FaceDetection(
	    	model_selection=1, min_detection_confidence=0.5) as face_detection:
			results = face_detection.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
			(h, w) = frame.shape[:2]
			# loop over the face detections
			if results.detections:
				for detection in results.detections:
					coordinates=detection.location_data.relative_bounding_box
					(startX, startY, endX, endY) = int(coordinates.xmin*w-10),int(coordinates.ymin*h-10),\
												   int((coordinates.width+coordinates.xmin)*w+10),\
												   int((coordinates.height+coordinates.ymin)*h+10)
					# ensure the bounding boxes fall within the dimensions of the frame
					(startX, startY) = (max(0, startX), max(0, startY))
					(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

					# extract the face ROI, convert it from BGR to RGB channel
					# ordering, resize it to 224x224, and preprocess it
					face = frame[startY:endY, startX:endX]
					# only make a predictions if at least one face was detected
					if face.any():
						face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
						face = cv2.resize(face, (224, 224))
						face = image_utils.img_to_array(face)
						face = preprocess_input(face)
						face = np.expand_dims(face, axis=0)
						# add the face and bounding boxes to their respective lists
						faces.append(face)
						locs.append((startX, startY, endX, endY))
						pred = maskNet.predict(face).tolist()[0]
						#check prediction and apply the argmax() to extract the corresponding class label
						pred_actual=np.argmax(pred)
						if pred_actual==0:
							mask="incorrect Mask"
						elif pred_actual==1:
							mask="Mask"
						elif pred_actual==2:
							mask="No Mask" 
						else:
							mask=""
						preds_actual.append(mask)
						preds.append(pred)
		return (locs, preds,preds_actual)
	

