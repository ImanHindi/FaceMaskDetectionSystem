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

	print("[INFO] loading face detector model...")
	faceNet = mp.solutions.face_detection
	mp_drawing = mp.solutions.drawing_utils
	
	
	
	#faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
	#faceNet = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	
	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	#maskNet = load_model(args["model"])
	maskNet = load_model('C:\\Users\\user\\Desktop\\iman\\FaceMaskDetection-SocialDistancing\\Model\\VGG19_FaceMaskDetector.hdf')
	#maskNet = load_model('C:\\Users\\user\\Desktop\\iman\\Face-Mask-Detection\\mask_detector.model')
	
	
	
	def detect_and_predict_mask(frame, faceNet=faceNet, maskNet=maskNet):
		faces = []
		locs = []
		preds = [] 
		preds_actual=[] 
		with faceNet.FaceDetection(
	    	model_selection=1, min_detection_confidence=0.5) as face_detection:
			#print(frame)
			#cv2.imshow("frame",frame)
			#cv2.waitKey()
			#cv2.destroyAllWindows()
	  		# Convert the BGR image to RGB and process it with MediaPipe Face Detection.
			
			results = face_detection.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
			#annotated_image = frame.copy()
			(h, w) = frame.shape[:2]
			#print(results.detections)
			#for detection in results.detections:
			#
			#	mp_drawing.draw_detection(annotated_image, detection)
			#	print(type(detection))
			#	cv2.imshow("annotated_image",annotated_image)
			#	cv2.waitKey()
			#	cv2.destroyAllWindows()
			#	cv2.imwrite('/annotated_image' + str(detection) + '.png', annotated_image)
	
			# loop over the detections
			if results.detections:
				for detection in results.detections:
					#print(detection.location_data.relative_bounding_box)
					coordinates=detection.location_data.relative_bounding_box
					#print(detection["location_data"]["relative_bounding_box "])
					#print(detection.values)
					#box = results.detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = int(coordinates.xmin*w-5),int(coordinates.ymin*h-5),int((coordinates.width+coordinates.xmin)*w+5),int((coordinates.height+coordinates.ymin)*h+5)#.location_data.relative_bounding_box
					# ensure the bounding boxes fall within the dimensions of
					# the frame
					#print(startX, startY, endX, endY)
					(startX, startY) = (max(0, startX), max(0, startY))
					(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
					#print(startX, startY, endX, endY)

					# extract the face ROI, convert it from BGR to RGB channel
					# ordering, resize it to 224x224, and preprocess it
					face = frame[startY:endY, startX:endX]
					#cv2.imshow("face",face)
					#cv2.waitKey()
					#cv2.destroyAllWindows()
					if face.any():
						face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
						#cv2.imshow("face",face)
						#cv2.waitKey()
						#cv2.destroyAllWindows()
						face = cv2.resize(face, (224, 224))
						face = image_utils.img_to_array(face)
						face = preprocess_input(face)
						face = np.expand_dims(face, axis=0)

						# add the face and bounding boxes to their respective
						# lists
						faces.append(face)
						locs.append((startX, startY, endX, endY))
					# only make a predictions if at least one face was detected
					#if len(faces) > 0:
						# for faster inference we'll make batch predictions on *all*
						# faces at the same time rather than one-by-one predictions
						# in the above `for` loop
						#faces = np.array(faces, dtype="float32")
						pred = maskNet.predict(face).tolist()[0]
						#(incorrect_Mask,mask, withoutMask)
						#if prediction.argmax()==0:
						#	pred="mask"
						#elif prediction.argmax()==1:
						#	pred="withoutMask"
						#elif prediction.argmax()==2:
						#	pred="incorrect_Mask"
						#print(pred)
					
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
						#print("detector",locs,preds)
			#else:
			#	preds=[0,0,0]
			#	locs=[0,0,0]

		return (locs, preds,preds_actual)
	#im=cv2.imread("C:\\Users\\user\\Desktop\\iman\\English\\lightness1.jpg")
	##im=cv2.resize(im,(500,500))
	#cv2.imshow("im",im)
	#cv2.waitKey()
	#cv2.destroyAllWindows()
	#detect_and_predict_mask(im)