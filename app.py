import os
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow.keras as keras
import shutil
import time
import pandas as pd

from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify
# from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER']="images"

model_loc = '../Paddy-Leaf-Disease-Detection-ML-Model/Code/MobileNetV2_Model-paddy-disease-detections.h5'
model = tf.keras.models.load_model(model_loc, custom_objects={'KerasLayer':hub.KerasLayer})
# model = tf.keras.models.load_model("paddy_leaf_disease_detection_model.h5", custom_objects={'KerasLayer':hub.KerasLayer})
# model = tf.keras.models.load_model("simple_model.h5")

# Server test functions
@app.route("/")
def hello():
    return 'Hello World'

def prepare_image(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)

# Machine Learning Predict and
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        shutil.rmtree('images')
        os.makedirs('images')
        upload_image=request.files['images']
        filepath=os.path.join(app.config['UPLOAD_FOLDER'],upload_image.filename)
        upload_image.save(filepath)

        #path ke gambar+nama filenya
        fname = "images/{}".format(os.listdir('images/')[0])
        df = pd.read_csv("model/label.csv", sep = ";")

        # Export Decoded Number
        def return_label(array):
            largest = 0
            print(array)
            for x in range(0, len(array)):
                if(array[x] > largest):
                    largest = array[x]
                    y = x
            return y
        
        # Read the image V1
        preprocessed_image = prepare_image(fname)
        result = model.predict(preprocessed_image)
        label = return_label(result[0])
        print(result)
        print(label)

        if label == 0:
            id = int(str(time.time()).replace('.', ''))
            label = 'Brown Spot'
            description = df.loc[df["Label"] == label]["Description"][0]
            solution = df.loc[df["Label"] == label]["Solution"][0]
        elif label == 1:
            id = int(str(time.time()).replace('.', ''))
            label = 'Healthy'
            description = df.loc[df["Label"] == label]["Description"][1]
            solution = df.loc[df["Label"] == label]["Solution"][1]
        elif label == 2:
            id = int(str(time.time()).replace('.', ''))
            label = 'Hispa'
            description = df.loc[df["Label"] == label]["Description"][2]
            solution = df.loc[df["Label"] == label]["Solution"][2]
        elif label == 3:
            id = int(str(time.time()).replace('.', ''))
            label = 'Leaf Blast'
            description = df.loc[df["Label"] == "Leaf Blast"]["Description"][3]
            solution = df.loc[df["Label"] == label]["Solution"][3]
        return jsonify(id=id, label=label, description=description, solution=solution)
    
    else: 
        print(KeyError)
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)