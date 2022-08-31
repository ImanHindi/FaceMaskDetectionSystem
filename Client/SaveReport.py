from cProfile import label
import traceback
from MaskDetectionRequest import get_report_request_from_firebase, get_report_request_from_localdb
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
import dataframe_image as dfi
from matplotlib import pyplot as plt



def json_to_csv_format(report):
	df=pd.json_normalize(report['result'])
	df.to_csv(f"x.csv")
	x=df.columns
	df=df.transpose()
	df["date"]=x
	out=df['date'].str.split('.',expand=True)	
	out['time']=out[1]+"."+out[3]+"."+out[5]
	out['pred']=out[6]
	out.drop([0,1,2,3,4,5,6], axis=1,inplace=True)
	out["output"]=df[0].values
	print("adding output",out)
	out.dropna( axis=0,inplace=True)
	try:
		out=out.pivot(index='time',columns='pred',values='output')
		out=out.explode(["facelocation","maskdetection","prediction"],ignore_index=False)
		print(out)
		out.style.format()
	except:
		pass
	df_styled2=out.head(10)
	dfi.export(df_styled2, "reportsample3.png")
	return out

	
def prepare_and_save_report(source=0):
	report=get_report_request_from_firebase(source)
	if report['result']:
		csv_format_report=json_to_csv_format(report)
		date_time=datetime.datetime.now()
		csv_format_report.to_csv(f"Report'{date_time.date()}'.csv")
		Summary = csv_format_report.groupby('maskdetection',as_index=False).count()
		Summary=pd.DataFrame(Summary)
		Summary.drop(labels='prediction',axis=1,inplace=True)
		print(Summary)
		try:
			mask_total=Summary.query('maskdetection == "Mask"')['facelocation'].iloc[0]	
		except:
			mask_total=0

		compliance_percentage=int((mask_total/Summary['facelocation'].sum())*100)
		compliance=np.array(Summary['facelocation'])
		labels=np.array(Summary['maskdetection'])
		plt.pie(compliance,labels=labels)
		plt.legend(title="compliance Pie Chart")
		plt.savefig('compliance Pie Chart')
		plt.show()
		
		Summary.loc[len(Summary.index)] = ['compliance_percentage',f"{compliance_percentage}%"]
		Summary.set_index('maskdetection',inplace=True)
		Styled_Summary=Summary.head(4).style.background_gradient(cmap='Blues')
		
		dfi.export(Styled_Summary, "Summarysample2.png")
		print(f"compliance_percentage={compliance_percentage}%")
		
		

	else:
		report=get_report_request_from_localdb(source)
		print(report)
		if (report['result']):
			csv_format_report=pd.DataFrame(report['result'],columns=["index","year","month","day","hour","minute","second","locations","predictions","maskdetection"])
			date_time=datetime.datetime.now()
			csv_format_report.to_csv(f"Report'{date_time.date()}'.csv")
			csv_format_report=csv_format_report.explode(["locations","maskdetection","predictions"],ignore_index=True).reset_index()

			Summary = csv_format_report.groupby('maskdetection',as_index=False).count()

			Summary=pd.DataFrame(Summary)
			Summary.drop(labels=["level_0","index","year","month","day","hour","minute","second","predictions"],axis=1,inplace=True)
			print(Summary)
			try:
				mask_total=Summary['locations'].iloc[0]
				print(mask_total)
			except:
				#traceback.print_exc()
				mask_total=0

			compliance_percentage=int((mask_total/Summary['locations'].sum())*100)


			Summary.loc[len(Summary.index)] = ['compliance_percentage',f"{compliance_percentage}%"]
			Summary.set_index('maskdetection',inplace=True)
			Styled_Summary=Summary.head(4).style.background_gradient(cmap='Blues')

			dfi.export(Styled_Summary, "Summarysample2.png")
			print(f"compliance_percentage={compliance_percentage}%")
