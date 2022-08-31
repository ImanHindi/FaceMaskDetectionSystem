#python RealTimeMaskDetection.py

# import the necessary packages
import datetime
import json
import queue
import threading
from tokenize import Ignore
from imutils.video import VideoStream
import pandas as pd
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from MaskDetectionRequest import prediction_request
from SaveReport import prepare_and_save_report

class VideoCapture:

	def __init__(self, name):
		print("[INFO] starting video stream...")
		self.cap = cv2.VideoCapture(name)
		self.q = queue.Queue()
		t = threading.Thread(target=self._reader)
		t.daemon = True
		t.start()

	# read frames as soon as they are available, keeping only most recent one
	def _reader(self):
		while True:
			ret, frame = self.cap.read()
			if not ret:
				break
			if not self.q.empty():
				try:
					self.q.get_nowait()   # discard previous (unprocessed) frame
				except queue.Empty:
					pass
			self.q.put(frame)
	
	def read(self):
		return self.q.get()

	# initialize the video stream and allow the camera sensor to warm upwdd
cap = VideoCapture(0)	
# loop over the frames from the video stream
while True:
	frame = cap.read()
	frame = imutils.resize(frame, width=400)
	cv2.imwrite('imag.png',frame)
	# send detection request
	(locs, preds,preds_actual) = prediction_request('imag.png')
	# loop over the detected face locations and their corresponding detections
	if (locs, preds):
		for (box, pred,pred_actual) in zip(locs, preds,preds_actual):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
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
			# display the label and bounding box rectangle on the output frame
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if chr(cv2.waitKey(1)&255) == 'q':
		break
# do a bit of cleanup
cv2.destroyAllWindows()
prepare_and_save_report()


