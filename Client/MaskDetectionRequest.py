from datetime import time
import datetime
from email.mime import image
from enum import unique
from json import load,dump
from pickle import EMPTY_LIST
from queue import Empty
from threading import Thread
from time import sleep
from tkinter import Frame
import traceback
from unittest import result
from weakref import ref
import requests
import cv2
from itertools import count
#from APIs.SQLiteAPI import LocalDataBase
import pandas as pd
import numpy as np
from flask import Flask,request, jsonify
from keras.preprocessing.image import image_utils
import json   
import requests
import base64



def prepare_data_for_firebase(data,pred_datetime):
    
    ref={
        "master":"/MaskDetectionReport",
        "year":"/year/"+str(pred_datetime.year),
        "month":"/month/"+str(pred_datetime.month),
        "day":"/day/"+str(pred_datetime.day),
        "hour":"/hour/"+str(pred_datetime.hour),
        "minute":"/minute/"+str(pred_datetime.minute),
        "second":"/second/"+str(pred_datetime.second)
    }
    json_dic={
        "facelocation":data['locs'],
        "prediction":data['preds'],
        "maskdetection":data['preds_actual']
    }
    return ref,json_dic

def firebase_task(data,pred_datetime):
    
    url ="http://localhost:5001/InsertData"
    print("threadtask",data)
    if any(data.values()):
        print("nullpart",data['locs'])
        ref,json_dic=prepare_data_for_firebase(data,pred_datetime)
        
        json_dic=json.dumps({'json_dic':json_dic,'ref':ref})
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    
        res=requests.post(url,data=json_dic,headers=headers)
        print(res)  
    
def SQlite3_task(data,pred_datetime):
    if any(data.values()):
        json_dic={
        "db":"MaskDetection",
        "TableName":"MaskDetectionOutput",
        "Columns":"ID INTEGER PRIMARY KEY AUTOINCREMENT ,year varchar(255) ,month varchar(255),day varchar(255),hour varchar(255),minute varchar(255) ,second varchar(255),locations BLOB,predictions BLOB,maskdetection BLOB",
        }
        url ="http://localhost:5002/CreateTable"
        #json_dic=json.dumps({'json_dic':json_dic})
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        
        res=requests.post(url,json=json_dic,headers=headers)
        print(res)
        if data['locs'] and data['preds']:    
            json_dic={
            "db":"MaskDetection",
            "TableName":"MaskDetectionOutput",
            "Columns":"year ,month ,day ,hour ,minute  ,second ,locations ,predictions,maskdetection ",
            "Values": {
                       "year": pred_datetime.year,
                      "month":pred_datetime.month,
                      "day":pred_datetime.day,
                      "hour":pred_datetime.hour,
                      "minute":pred_datetime.minute,
                      "second":pred_datetime.second,
                      "locations":data['locs'],
                      "predictions":data['preds'],
                      "maskdetection":data['preds_actual']
            }
            }

            url ="http://localhost:5002/InsertData"
            #json_dic=json.dumps({'json_dic':json_dic})
            headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
            res=requests.post(url,json=json_dic,headers=headers)
            print(res)






def prediction_request(frame):
    #Web Service Prediction:
    url ="http://127.0.0.1:5000/PredictMask"
    try:
        with open(frame, "rb") as f:
            im_bytes = f.read()        
        im_b64 = base64.b64encode(im_bytes).decode("utf8")    
        json_im = json.dumps({'image': im_b64})
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

        response = requests.post(url, data=json_im,headers=headers)
        #print (response)
        data = response.json()
        print(data)
        #firebase_task(data)
        #SQlite3_task(data)
        
        pred_datetime=datetime.datetime.now()
        if pred_datetime.hour==None:
            pred_datetime.hour=0
        if pred_datetime.minute==None:
            pred_datetime.minute=0
        if pred_datetime.second==None:
            pred_datetime.second=0
        global firebase_thread
        global SQlite3_thread
        
        firebase_thread= Thread(target=firebase_task,args=(data,pred_datetime))
        SQlite3_thread= Thread(target=SQlite3_task,args=(data,pred_datetime))
        firebase_thread.start()
        SQlite3_thread.start()
        
        return data['locs'],data['preds'],data['preds_actual']

    except:
        print( "check connection to the server and try again..")
        return 
        

   
def get_report_request_from_firebase():
    
    pred_datetime=datetime.datetime.now()
    
    try:
        firebase_thread.join()
        ref={
            "master":"/MaskDetectionReport",
            "year":"/year/"+str(pred_datetime.year),
            "month":"/month/"+str(pred_datetime.month),
            "day":"/day/"+str(pred_datetime.day),
        }
        json_dic=json.dumps({'ref':ref})          
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

        url ="http://localhost:5001/GetbyDate"
        response = requests.get(url, data=json_dic,headers=headers)
        response=json.loads(response.text)
        print(response,"this is request report response")
    except:
        print("cannot retreive the report... check internet connection and try again.")
        response=json.loads({'result':0})
    return response

def get_report_request_from_localdb():
    try:
        pred_datetime=datetime.datetime.now()

        json_dic={
            "db":"MaskDetection",
            "TableName":"MaskDetectionOutput",
            "Columns":{"year": "year",
                      "month":"month" ,
                      "day": "day"},
            "Values": {
                       "year": pred_datetime.year,
                      "month":pred_datetime.month,
                      "day":pred_datetime.day,

            }
            }
        SQlite3_thread.join()
        url ="http://localhost:5002/GetbyDate"
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        response = requests.get(url, json=json_dic,headers=headers)
        print("response from local db in detection request",response)
        
        print(response.text['result'])
        response=json.loads(response.text)
        
    except:
        traceback.print_exc()
        print("retreiving report.... please keep connection on...")
        response=json.loads({'result':0})
    return response




