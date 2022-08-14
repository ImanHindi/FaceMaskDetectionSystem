from datetime import time
import datetime
from email.mime import image
from json import load,dump
from time import sleep
import requests
import cv2
from APIs.SQLiteAPI import LocalDataBase
import pandas as pd
import numpy as np
from Client.RealTimeMaskDetection import single_shot, start_testing
from flask import Flask,request, jsonify

datetime=[]
def prediction_request(frame):

    while True:
        sleep(2)
        image_shot=frame
        image_shot=bytearray(image_shot)
        #Web Service Prediction:
        url ="http://localhost:5000/PredictMask"
        (locs,preds) = requests.post(url, image = image_shot)
        #Face_locations=str(locs)
        #predictions=str(preds)
        pred_datetime=datetime.datetime.now()
        year=pred_datetime.year
        month=pred_datetime.month
        day=pred_datetime.day
        hour=pred_datetime.hour
        minute=pred_datetime.minute
        
        json_dic={
        "db":"MaskDetection.db",
        "TableName":"MaskDetectionOutput.db",
        "CoLoums":"ID,Date,Time,pred",
        }
        url ="http://localhost:5000/localdb/CreateTable"
        res=requests.post(url,json=json_dic)
        
        json_dic={
        "db":"MaskDetection.db",
        "TableName":"MaskDetectionOutput.db",
        "CoLoums":"ID,Date,Time,pred",
        "Values": str(year)+str(month)+str(day)+str(hour)+str(minute)+str(locs)+str(preds)
        }
        
        url ="http://localhost:5000/localdb/InsertData"
        res=requests.post(url,json=json_dic)
        json_dic={
            "year": year, 
            "month":month,
            "day":day,
            "hour":hour,
            "minute":minute,
            "facelocation":locs,
            "prediction":preds
        }
        url ="http://localhost:5000/InsertData"
        res=requests.post(url,json=json_dic)

        return locs,preds


        ##output: '{"MaskPrediction":['WithMask','WithOutMask','IncorrectMask']}'    



data=local_database.get_data()
df=pd.DataFrame(data)
print(df)
df.to_csv("Report.csv")