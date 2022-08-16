# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from MaskDetectionRequest import prediction_request


# initialize the video stream and allow the camera sensor to warm upwdd
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	#cv2.imshow("image",frame)
	#cv2.waitKey()
	#cv2.destroyAllWindows()
	cv2.imwrite('imag.png',frame)
	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = prediction_request('imag.png')

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
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
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
