from concurrent.futures import thread
from datetime import datetime
import json
import queue
from threading import Thread
from time import sleep
import traceback
import pandas as pd
from flask import Flask,request, jsonify
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from threading import Event

app = Flask('app')
q=queue.Queue()
q_ref=queue.Queue()
global stop_threads
def stop_threads_fun():
    stop_threads=True   
    return stop_threads
def q_thread_task():
    while True:
        
        if not q.empty():
            print("inter thread2")
            try:
              conn=FirebaseDataBase()
              if conn:
                print("q_ref","q")
                data=q.get()
                ref=q_ref.get()
                conn.insert_data(ref=ref,data=data)  
                print("insert queued data successfully")
                conn.close_connection()
                if q.empty():
                    print("out of thread")
                    break
            except:
              print("queue thread is on")
              continue
        elif stop_threads:
            break
        if stop_threads:
            break

           
        
            
class FirebaseDataBase:
    def __init__(self):
        self.conn=self.connect_to_database('https://face-mask-detection-2778f-default-rtdb.firebaseio.com/')
#function to make the connection to firebase   
    def connect_to_database(self,databaseURL):
        try:
            self.cred = credentials.Certificate('C:\\Users\\user\\Desktop\\iman\\FinalProject\\APIs\\face-mask-detection-2778f-firebase-adminsdk-qbbbj-95171a0d30.json')
            conn=firebase_admin.initialize_app(self.cred,{'databaseURL' : f'{databaseURL}', 'httpTimeout' : 30})
            print('successfull')
            return conn
        except:
            return "checking internet connection..."        
            
#insert data to firebase
    def insert_data(self, ref,data):
        try:
                
            root = db.reference(ref["master"]+ref["year"]+ref["month"]+ref["day"]+ref["hour"]+ref["minute"]+ref["second"])
            root.set(data)
            print("insert root res")
            return "successfully inserted"
        except:
            
            q.put(data)
            print(q)
            q_ref.put(ref)
            print(q_ref)
            #q_thread.start()
            return "checking internet connection..."
#get all data from firebase db
    def get_data(self, ref):
            root = db.reference(ref)
            try:
                data=root.get()
                return data
            except:
                return "checking internet connection..."
#retreive the data of specific day 
    def get_data_by_Date(self, ref): 
            
            root = db.reference(ref["master"]+ref["year"]+ref["month"]+ref["day"])
            print(root)
            try:
                data=root.get()
                print("data in firebase", data)
                result=json.dumps({"result":data})
                return result
            except:
                #traceback.print_exc()
                print("no connection to retreive the data.")
#delete data for a specific day
    def Delete_by_Date(self, ref):    
                root = db.reference(ref)
                try:
                    res=root.delete()
                    return res
                except:
                   # traceback.print_exc()
                    "checking internet connection..."
#close the connection to db 
    def close_connection(self):
        try:
            return firebase_admin.delete_app()
        except:
          #traceback.print_exc()
          return "checking internet connection..."
  


#Fetch All DB Data rout
@app.route('/Get_Data',methods=['GET'])         
def Get_Data():
    conn=FirebaseDataBase()
    ref=request.args.get("ref")
    
    data=conn.get_data(ref)
    conn.close_connection()
    return data
    

#get Data for a specified Date rout   
@app.route('/GetbyDate',methods=['GET'])           
def GetbyDate():
    
    q_thread.join()
    stop_threads_fun()
    q_thread.join()
    print("GetbyDate")
    conn=FirebaseDataBase()
    
    ref =request.json['ref']
    print(ref)
    data=conn.get_data_by_Date(ref)
    conn.close_connection()
    return data
    

#delete from DB rout
@app.route("/Delete",methods=['DELET'])
def Delete():
    conn=FirebaseDataBase()
    ref = str(request.args.get("ref"))
    res=conn.Delete_by_Date(ref)
    conn.close_connection()
    return res
    

#Insert new item to DB reout
@app.route('/InsertData',methods=['POST','GET'])
def InsertData():
        try: 
            print("InsertData")
            conn=FirebaseDataBase()
            ref =request.json['ref']
            print('ref',request.json['ref'])
            data=request.json['json_dic']
            print("firebasepart",data)
            #while not q.empty():
                #queued_data=json.dumps(q.get())
            #    res=conn.insert_data(q_ref,q) 
                
            conn.insert_data(ref,data) 
            print("successfully inserted")
            conn.close_connection()
            res= "successfully inserted"       
        except:
           
            res="checking internet connection..."
            
        return res




if __name__ == '__main__':
    try:
        stop_threads=False
        q_thread= Thread(target=q_thread_task,args=())
        q_thread.start()
        
        app.run(debug=True, host='0.0.0.0', port=5001)
        data=dict
        ref=""
        event = Event()
        
    except:
        print("checking server connection...")
