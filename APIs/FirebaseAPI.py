from datetime import datetime
import json
import queue
import traceback
import pandas as pd
from flask import Flask,request, jsonify
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

app = Flask('app')
q=queue.Queue()

class FirebaseDataBase:
    def __init__(self):
        self.conn=self.connect_to_database('https://face-mask-detection-2778f-default-rtdb.firebaseio.com/' 
)

    def connect_to_database(self,databaseURL):
        try:
            self.cred = credentials.Certificate('C:\\Users\\user\\Desktop\\iman\\FinalProject\\APIs\\face-mask-detection-2778f-firebase-adminsdk-qbbbj-95171a0d30.json')
            conn=firebase_admin.initialize_app(self.cred,{'databaseURL' : f'{databaseURL}', 'httpTimeout' : 30})
            return conn
        except:
            return traceback.print_exc()        
            

    def create_table(self, ref, tbname):
        try:
            tbname=json.loads(tbname)
            root = db.reference(ref)
            res=root.set(tbname)
            return res
        except:
            traceback.print_exc()
            return traceback.print_exc()

    def insert_data(self, ref,data):
        try:
            root = db.reference(ref)
            res=root.set(data)
            return res
        except:
            traceback.print_exc()
            return traceback.print_exc()

    def get_data(self, ref):
            ref=ref
            root = db.reference(ref)
            try:
                data=root.get()
                return data
            except:
                traceback.print_exc()
                return traceback.print_exc()

    def get_data_by_Date(self, ref):    
            ref=ref
            root = db.reference(ref)
            try:
                data=root.get()
                return data
            except:
                traceback.print_exc()
                return traceback.print_exc()

    def Delete_by_Date(self, ref):    
                ref=ref
                root = db.reference(ref)
                try:
                    res=root.delete()
                    return res
                except:
                    traceback.print_exc()
                    return traceback.print_exc()

    
    def close_connection(self):
        try:
            return firebase_admin.delete_app()
        except:
          traceback.print_exc()
          return traceback.print_exc()
  


#Fetch All DB Data
@app.route('/Get_Data',methods=['GET'])         
def Get_Data():
    conn=FirebaseDataBase()
    ref=request.args.get("ref")
    
    data=conn.get_data(ref)
    conn.close_connection()
    return data
    

#get Data for a specified Date      
@app.route('/GetbyDate',methods=['GET'])           
def GetbyDate():
    conn=FirebaseDataBase()
    ref=str(request.args.get("ref")+"/"+str(request.args.get("Date")+"/"))
    data=conn.get_data_by_Date(ref)
    conn.close_connection()
    return data
    

#delete from DB
@app.route("/Delete",methods=['DELET'])
def Delete():
    conn=FirebaseDataBase()
    ref = str(request.args.get("ref"))
    res=conn.Delete_by_Date(ref)
    conn.close_connection()
    return res


#create tables
@app.route('/CreateTable',methods=['POST','GET'])
def CreateTable():
    conn=FirebaseDataBase()
    ref =str(request.args.get("ref"))
    tb_name=request.args.get("tb_name")
    res=conn.create_table(ref,tb_name)
    conn.close_connection()
    return res

     

#Insert new item to DB
@app.route('/Insert_Data',methods=['POST','GET'])
def Insert_Data():
    try:

        conn=FirebaseDataBase()
        ref =str(request.args.get("ref"))
        data=request.args.get("data")
        data=json.dumps(data) 
        while not q.empty():
            queued_data=json.dumps(q.get())
            res=conn.insert_data(ref,queued_data) 
       
        res=conn.insert_data(ref,data) 
        conn.close_connection()
    except:
        q.put(data)

    return res
     









if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except:
        traceback.print_exc()
        

#'https://face-mask-detection-2778f-default-rtdb.firebaseio.com/' 