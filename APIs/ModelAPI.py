import pickle
from flask import Flask,request, jsonify
from Model.FaceMaskDetectionModel import DetectMask
from PIL import Image
import numpy as np

from Model.FaceMaskDetector import detect_and_predict_mask

app = Flask('app')




@app.route('/Prediction', methods=['GET'])
def prediction():
    return 'Pinging Model Application!!'

@app.route('/PredictMask', methods=['POST'])
def predict_mask():
    image = request.args.get("image")
    image=np.array(image)
    #file={'image': open(image,'rb')}
    #img=np.array(image.open(file.stream))

    print(image)
    
    (locs, preds) = detect_and_predict_mask(image)
    
    

    result = {
        'MaskPrediction': list(preds)
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)