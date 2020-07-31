import os
from flask import Flask, render_template, request
from flask import send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = dir_path + '/uploads'
STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

graph = tf.get_default_graph()
with graph.as_default():
    # load model at very first
   model = load_model('cancer.h5')


# call model to predict an image
def api(full_path):
    data = image.load_img(full_path, target_size=(64, 64))
    data = image.img_to_array(data)
    data = np.expand_dims(data, axis=0)
    #data = data * 1.0 / 255

    with graph.as_default():
        predicted = model.predict(data)
        return predicted


# home page
@app.route('/')
def home():
   return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = 'uploads/' + file.filename
        file.save(full_name)

        #indices = {0: 'Dog', 1: 'Cat', 2: 'Invasive carcinomar', 3: 'Normal'}
        result = api(full_name)

        predicted_class = np.asscalar(np.argmax(result, axis=1))
        accuracy = round(result[0][predicted_class] * 100, 2)

        if result[0][0] >= 0.5:
          label = 'This image may not contains any symptoms of skin cancer'
        else:
          label = 'This image may contains some symptoms of skin cancer'

    return render_template('predict.html', image_file_name = file.filename, label = label)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True

