from datetime import datetime
import json
import sqlite3
import traceback
import pandas as pd
from flask import Flask,request

app = Flask('app')


class LocalDataBase:
    def __init__(self, dbname):
        try:
            self.conn = sqlite3.connect(f"{dbname}.db")
            self.cursor = self.conn.cursor()
            self.cursor.execute("PRAGMA journal_mode=WAL")
            self.conn.commit()
        except:
            print("checking localdb connection...")
    
#creat a table on local db
    def create_table(self, tbname, columns):
        query=f"CREATE TABLE IF NOT EXISTS '{tbname}'({columns});"
        try:
            self.cursor.execute(query)
            self.conn.commit()
            return "created successfully"
        except:
            traceback.print_exc()
            return "checking localdb connection..."
#insert data to local db
    def insert_data(self, tbname, columns, values):
        print('insert')
        query=f'''INSERT INTO '{tbname}' ( {columns} ) 
                VALUES ({values['year']},
                {values['month']},{values['day']},
                {values['hour']},{values['minute']},
                {values['second']},"{values['locations']}",
                "{values['predictions']}","{values['maskdetection']}"  );'''
        try:
            self.cursor.execute(query)
            self.conn.commit()
            return "inserted successfully"
        except:
            traceback.print_exc()
            return "checking localdb connection..."
#get all data from local db
    def get_data(self, tbname,colum,value):
        query=f"SELECT * FROM {tbname} WHERE {colum['year']} = {value['year']} AND {colum['month']} = {value['month']} AND {colum['day']} = {value['day']} "
        try:
            data=self.cursor.execute(query).fetchall()
            self.conn.commit()
            print("getting",data)
            return data
        except:
            traceback.print_exc()
            return "checking localdb connection..."
#close the connection
    def close_connection(self):
        self.conn.close()

#routs and fuction of SQlite API
@app.route('/CreateTable', methods=['POST','GET'])
def CreateTable():
    query=request.json
    print(query)
    DataBase = query['db']
    print(DataBase)
    local_database = LocalDataBase(DataBase)
    Table= query['TableName']
    Columns=query['Columns']
    result =local_database.create_table(Table,Columns)
    local_database.close_connection()
    return result

@app.route('/InsertData', methods=['POST','GET'])
def InsertData():
    query=request.json
    print(query)
    DataBase = query['db']
    print(DataBase)
    local_database = LocalDataBase(DataBase)
    Table= query['TableName']
    Columns=query['Columns']
    Values=query['Values']
    print(Table,Columns,Values)
    result =local_database.insert_data(Table,Columns,Values)

    local_database.close_connection()
    return "successfully added"

@app.route('/GetbyDate', methods=['POST','GET'])
def GetbyDate():
    query=request.json
    DataBase = query['db']
    local_database = LocalDataBase(DataBase)
    Table= query['TableName']
    Columns=query['Columns']
    Values=query['Values']
    
    result =local_database.get_data(Table,Columns,Values)
    local_database.close_connection()
    result=json.dumps({"result": result})
    print(result,"out result")
    return result


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)