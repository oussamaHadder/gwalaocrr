from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from GwalaOCR import GwalaOCR
import os
from models.ocr_predictor import get_predictor
from flask_cors import CORS
import numpy as np
import cv2
import json

app = Flask(__name__)
CORS(app)
api = Api(app=app)

predictor = get_predictor()
model_path = os.path.join(app.root_path, 'models', 'FinetuneV3.2')

class ExtractInfo(Resource):
    def post(self):
        # Check if the POST request contains a file
        if 'file' not in request.files:
            return {'error': 'No file part'}, 400

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return {'error': 'No selected file'}, 400

        # Save the uploaded file to a temporary location
        file_path = os.path.join(app.root_path, 'temp', file.filename)
        file.save(file_path)

        # Process the image and return extracted data
        gwalaocr = GwalaOCR(predictor=predictor, model_path=model_path)
        json_data = gwalaocr.process_receipt_image(file_path)


        os.remove(file_path)


        json_data = json.loads(json_data)


        return json_data

api.add_resource(ExtractInfo, "/extract")

if __name__ == "__main__":
    app.run()
