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

class FirebaseDataBase:
    def __init__(self):
        self.conn=self.connect_to_database('https://face-mask-detection-2778f-default-rtdb.firebaseio.com/')
        

    def connect_to_database(self,databaseURL):
        try:
            self.cred = credentials.Certificate('C:\\Users\\user\\Desktop\\iman\\FinalProject\\APIs\\face-mask-detection-2778f-firebase-adminsdk-qbbbj-95171a0d30.json')
            conn=firebase_admin.initialize_app(self.cred,{'databaseURL' : f'{databaseURL}', 'httpTimeout' : 30})
            print('successfull')
            return conn
        except:
            return "checking internet connection..."        
            

    #def create_table(self, ref, tbname):
    #    try:
    #        tbname=json.loads(tbname)
    #        root = db.reference(ref)
    #        res=root.set(tbname)
    #        return res
    #    except:
    #        traceback.print_exc()
    #        return traceback.print_exc()

    def insert_data(self, ref,data):
        try:
                
            root = db.reference(ref["master"]+ref["year"]+ref["month"]+ref["day"]+ref["hour"]+ref["minute"]+ref["second"])
            #res=root.set(data)
            #root.parent
            #rootReference = db.child("-Blah1").child("name").getRoot();
            #db.parse
            res=root.set(data)
            #root=root.child(data)
            #res=root.set(data['year']['month']['day']['hour']['minute']['second'])
            return res
        except:
            traceback.print_exc()
            return "checking internet connection..."

    def get_data(self, ref):
            root = db.reference(ref)
            try:
                data=root.get()
                return data
            except:
                traceback.print_exc()
                return "checking internet connection..."

    def get_data_by_Date(self, ref):    
            
            root = db.reference(ref["master"]+ref["year"]+ref["month"]+ref["day"])
            try:
                data=root.get()
                print("data in firebase", data)
                result=json.dumps({"result":data})
                return result
            except:
                traceback.print_exc()
                return traceback.print_exc()

    def Delete_by_Date(self, ref):    
                root = db.reference(ref)
                try:
                    res=root.delete()
                    return res
                except:
                    traceback.print_exc()
                    "checking internet connection..."

    
    def close_connection(self):
        try:
            return firebase_admin.delete_app()
        except:
          traceback.print_exc()
          return "checking internet connection..."
  


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
    print("GetbyDate")
    conn=FirebaseDataBase()
    
    ref =request.json['ref']
    print(ref)
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



     

#Insert new item to DB
@app.route('/InsertData',methods=['POST','GET'])
def InsertData():
    
        print("InsertData")
        conn=FirebaseDataBase()
        #ref=request.args.get('ref')
        ref =request.json['ref']
        print('ref',request.json['ref'])
        data=request.json['json_dic']
        print("firebasepart",data)
        #data=json.dumps(data) 
        while not q.empty():
            queued_data=json.dumps(q.get())
            res=conn.insert_data(ref,queued_data) 
        #thread1= Thread(target=task,args=(data,ref))
        #thread1.start()
        try:  
            res=conn.insert_data(ref,data) 
            print( "successfully inserted")
            conn.close_connection()
            return "successfully inserted"
            
        except:
            q.put(ref,data)
            print("checking internet connection...")
            
            
        return res
     


#def task(data,ref):
#    conn=FirebaseDataBase()
#    if conn.connect_to_database():
#        while not q.empty():
#            (ref,data)=json.dumps(q.get())
#            res=conn.insert_data(ref,data)
#            print(res)
#            
#        conn.close_connection()
#        
#    else:
#        q.put(ref,data)
#    # display a message
    
#    print('This is from another thread')
    






if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
        data=dict
        ref=""
        event = Event()

    except:
        traceback.print_exc()
        

#'https://face-mask-detection-2778f-default-rtdb.firebaseio.com/' 