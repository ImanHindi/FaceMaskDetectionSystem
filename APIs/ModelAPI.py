import base64
import json
import logging
import pickle
from typing import IO
from flask import Flask,request, jsonify,abort
from PIL import Image
import numpy as np
import io

#from FaceMaskDetector import detect_and_predict_mask
import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'Model')
sys.path.append( mymodule_dir )
import FaceMaskDetector  

app = Flask('app')
app.logger.setLevel(logging.DEBUG)




@app.route('/Prediction', methods=['GET'])
def prediction():
    return 'Pinging Model Application!!'

@app.route('/PredictMask', methods=['POST','GET'])
def predict_mask():
    # print(request.json)      
    if not request.json or 'image' not in request.json: 
        abort(400)
    im_b64 = request.json['image']
    #json = bytearray(request.get_json())
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))
    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))
    img_arr = np.asarray(img)      
    #file={'image': open(image,'rb')}
    #img=np.array(image.open(file.stream))
    #print(image)
    locs, preds = FaceMaskDetector.FaceMaskDetector.detect_and_predict_mask(img_arr)
    #print(locs)
    #print(preds)
    
    result = {
        'locs': locs,

        'preds': preds
    }
    return result


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)