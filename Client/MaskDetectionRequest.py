from datetime import time
import datetime
from email.mime import image
from json import load,dump
from time import sleep
import requests
import cv2
#from APIs.SQLiteAPI import LocalDataBase
import pandas as pd
import numpy as np
from flask import Flask,request, jsonify
from keras.preprocessing.image import image_utils
import json   
import requests
import base64
def prediction_request(frame):

    #Web Service Prediction:
    
    url ="http://localhost:5000/PredictMask"
    #cv2.imshow("image_shot",frame)
    #cv2.waitKey()
    #cv2.destroyAllWindows()  
    #im_bytes = bytearray(frame)
    with open(frame, "rb") as f:
        im_bytes = f.read()        
    im_b64 = base64.b64encode(im_bytes).decode("utf8")    
    json_im = json.dumps({'image': im_b64})
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    response = requests.post(url, data=json_im,headers=headers)
    #print (response)
    data = response.json()
        #(locs,preds) = requests.post(url, json={'image':  str(image_shot)})
        
        #pred_datetime=datetime.datetime.now()
        #year=pred_datetime.year
        #month=pred_datetime.month
        #day=pred_datetime.day
        #hour=pred_datetime.hour
        #minute=pred_datetime.minute
        #
        #json_dic={
        #"db":"MaskDetection.db",
        #"TableName":"MaskDetectionOutput.db",
        #"CoLoums":"ID,Date,Time,pred",
        #}
        #url ="http://localhost:5000/localdb/CreateTable"
        #res=requests.post(url,json=json_dic)
        #
        #json_dic={
        #"db":"MaskDetection.db",
        #"TableName":"MaskDetectionOutput.db",
        #"CoLoums":"ID,Date,Time,pred",
        #"Values": str(year)+str(month)+str(day)+str(hour)+str(minute)+str(locs)+str(preds)
        #}
        #
        #url ="http://localhost:5000/localdb/InsertData"
        #res=requests.post(url,json=json_dic)
        #json_dic={
        #    "year": year, 
        #    "month":month,
        #    "day":day,
        #    "hour":hour,
        #    "minute":minute,
        #    "facelocation":locs,
        #    "prediction":preds
        #}
        #url ="http://localhost:5000/InsertData"
        #res=requests.post(url,json=json_dic)
        #location=response.text['locs']
        #predictions=response.text['preds']      
        
    return data['locs'],data['preds']


        ##output: '{"MaskPrediction":['WithMask','WithOutMask','IncorrectMask']}'    



#data=local_database.get_data()
#df=pd.DataFrame(data)
#print(df)
#df.to_csv("Report.csv")