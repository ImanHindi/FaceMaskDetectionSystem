from datetime import datetime
import sqlite3
import traceback
import pandas as pd
from flask import Flask,request

app = Flask('app')


class LocalDataBase:
    def __init__(self, dbname):
        self.conn = self.connect_to_database(dbname)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA journal_mode=WAL")
        self.conn.commit()

    def connect_to_database(self,dbname):
        try:
            conn=sqlite3.connect(f"{dbname}.db")
            return conn
        except:
            return traceback.print_exc()

    def create_table(self, tbname, columns):
        query=f"CREATE TABLE IF NOT EXISTS {tbname}({columns})"
        try:
            self.cursor.execute(query)
            self.conn.commit()
            return "created successfully"
        except:
            traceback.print_exc()
            return traceback.print_exc()

    def insert_data(self, tbname, columns, values):
        query=f"INSERT INTO {tbname}({columns}) VALUES ({values})"
        try:
            self.cursor.execute(query)
            self.conn.commit()
            return "inserted successfully"
        except:
            traceback.print_exc()
            return traceback.print_exc()

    def get_data(self, tbname,colum,value):
        query=f"SELECT * FROM {tbname} WHERE {colum} = {value} "
        try:
            data=self.cursor.execute(query).fetchall()
            self.conn.commit()
            return data
        except:
            traceback.print_exc()
            return traceback.print_exc()

    def close_connection(self):
        self.conn.close()


@app.route('localdb/CreateTable', methods=['POST'])
def CreateTable():
    DataBase = request.args.get("db")
    local_database = LocalDataBase(DataBase)
    Table= request.args.get("Table")
    Columns=request.args.get("Colum")
    result =local_database.create_table(Table,Columns)
    local_database.close_connection()
    return result

@app.route('localdb/InsertData', methods=['POST'])
def InsertData():
    json=request.get_json("json")
    DataBase = json["db"]
    local_database = LocalDataBase(DataBase)
    Table= json["Table"]
    Columns=json["Colums"]
    Values=json["Values"]
    result =local_database.insert_data(Table,Columns,Values)
    local_database.close_connection()
    return result

@app.route('localdb/GetReport', methods=['POST','GET'])
def GetReport():
    DataBase = request.args.get("db")
    local_database = LocalDataBase(DataBase)
    Table= request.args.get("Table")
    Column=request.args.get("Colum")
    Value=request.args.get("Value")
    
    result =local_database.get_data(Table,Column,Value)
    local_database.close_connection()
    return result



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)