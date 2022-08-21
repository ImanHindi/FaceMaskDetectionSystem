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
from MaskDetectionRequest import get_report_request_from_firebase, get_report_request_from_localdb

from MaskDetectionRequest import prediction_request
import csv

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
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = cap.read()
	frame = imutils.resize(frame, width=400)
	#cv2.imshow("image",frame)
	#cv2.waitKey()
	#cv2.destroyAllWindows()
	cv2.imwrite('imag.png',frame)
	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds,preds_actual) = prediction_request('imag.png')

	# loop over the detected face locations and their corresponding
	# locations
	if (locs, preds):
		for (box, pred,preds_actual) in zip(locs, preds,preds_actual):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			#(mask,incorrect_Mask, withoutMask) = pred
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
#cap.stop()
date_time=datetime.datetime.now()
report=get_report_request_from_firebase()
print(report['result'],'realtime')

if report['result']:
	
	df=pd.json_normalize(report['result'],record_path="second",meta=[['hour','minute','second']])
	#df=df[0].str.split('.', expand=True)
	#df=df.explode(df.columns[0]).reset_index(drop=True)
	#data = list(map(flatten_dict, report['result']))
	#df = pd.DataFrame(data)
	#report=json.loads(report["result"])
else:
	report=get_report_request_from_localdb()
	df=pd.DataFrame(report['result'])
print(df)
date_time=datetime.datetime.now()
df.to_csv(f"Report'{date_time.date()}'.csv")
pd.read_csv(f"Report'{date_time.date()}'.csv")

