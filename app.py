import os
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow.keras as keras

from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify
# from werkzeug.utils import secure_filename



app = Flask(__name__)

app.config['UPLOAD_FOLDER']="images"
# model = tf.keras.models.load_model("paddy_leaf_disease_detection_model.h5")
# model = tf.keras.models.load_model("paddy_leaf_disease_detection_model.h5", custom_objects={'KerasLayer':hub.KerasLayer})
model = tf.keras.models.load_model("simple_model.h5")

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        upload_image=request.files['image']
        
        filepath=os.path.join(app.config['UPLOAD_FOLDER'],upload_image.filename)
        upload_image.save(filepath)

        #path ke gambar+nama filenya
        fname = 'images/images.jpg'
        # model = tf.keras.models.load_model("paddy_leaf_disease_detection_model.h5", custom_objects={'KerasLayer':hub.KerasLayer})
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), 
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])

        # Read the image
        image_size = (244, 244)
        test_image = image.load_img(fname, target_size = image_size)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
    return  jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)