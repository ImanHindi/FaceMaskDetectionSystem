import base64
import json
import logging
import pickle
from typing import IO
from flask import Flask,request, jsonify,abort
from PIL import Image
import numpy as np
import io
import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'Model')
sys.path.append( mymodule_dir )
import FaceMaskDetector  

app = Flask('app')
app.logger.setLevel(logging.DEBUG)




# the rout to send the mask detection request to the server
@app.route('/PredictMask', methods=['POST','GET'])
def predict_mask(): 
    try:  
        if not request.json or 'image' not in request.json: 
            abort(400)
        im_b64 = request.json['image']
        img_bytes = base64.b64decode(im_b64.encode('utf-8'))
        # convert bytes data to PIL Image object
        img = Image.open(io.BytesIO(img_bytes))
        img_arr = np.asarray(img)

        locs, preds,preds_actual = FaceMaskDetector.FaceMaskDetector.detect_and_predict_mask(img_arr)
        #return the mask detection results as a dictionary

        result = {
            'locs': locs,
            'preds': preds,
            'preds_actual':preds_actual
        }
    
        return result
    except:
        print("Check server Connection and try again..")

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)