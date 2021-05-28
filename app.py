import os
from flask import Flask, render_template, request, flash
from flask_uploads import IMAGES, UploadSet, configure_uploads

app = Flask(__name__)
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"
app.config["SECRET_KEY"] = os.urandom(24)
configure_uploads(app, photos)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('upload.html')

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'image' in request.files:
        # Get the image 
        photos.save(request.files['image'])
        flash("Image saved successfully.")
        '''
        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
        '''
        # Return the view or Json
        return render_template('upload.html')
    return render_template('upload.html')

if __name__ == '__main__':
    # Setting with ip and port HTTP
    app.run(host='0.0.0.0', port=80, debug=True)