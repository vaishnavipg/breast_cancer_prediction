from __future__ import division, print_function
import os
import numpy as np
from keras. preprocessing import image
from keras.models import load_model
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

global graph
# graph = tf.get_default_graph()

app = Flask(__name__, template_folder='../templates')

model = load_model('breastcancer.h5')


@app.route("/", methods=['GET'])
def index():
    return render_template('bcancer.html')


@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # Get the image file from the request
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, secure_filename(f.filename))
        f.save(file_path)
        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(50, 50))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        preds = model.predict(x)

        app.logger.debug(np.round(preds[0][1],0))

        if np.round(preds[0][1],0) == 0.0:
            text = "The tumor is benign.. Need not worry!"    
        else:
            text = "It is a malignant tumor... Please Consult Doctor "
        text = text

        app.logger.debug(text)

        return text


if __name__ == '__main__':
    app.run(debug = True)

