import argparse
import io
from PIL import Image
import datetime

import torch
import cv2
import numpy as np
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
from werkzeug.utils import secure_filename

import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob


from ultralytics import YOLO

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST" and 'file' in request.files:
        f = request.files['file']
        if f.filename == '':
            return "No selected file"
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            filepath = os.path.join('upload', filename)
            f.save(filepath)

            file_extension = filename.rsplit(".", 1)[1].lower()
            if file_extension == 'jpg':
                img = cv2.imread(filepath)
                frame = cv2.imencode(".jpg", cv2.UMat(img))[1].tobytes()

                image = Image.open(io.BytesIO(frame))

                yolo = YOLO('yolov8n.pt')
                detect = yolo.predict(image, save=True)

                folder_path = './runs/detect'
                subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
                latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
                image_path = os.path.join(folder_path, latest_subfolder, filename)

                return render_template('index.html', image_path=image_path)
        else:
            return "Invalid file format"
    return render_template("index.html")

@app.route('/<filename>')
def display(filename):
    folder_path = './runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = os.path.join(folder_path, latest_subfolder)

    return send_from_directory(directory, filename)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="flask app")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    app.run(port=args.port)
