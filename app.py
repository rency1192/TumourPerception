from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('templates/my_model.h5')

# Define the labels for the classes
class_names = ['Benign', 'Malignant']


# index.html page
@app.route('/')
def index():
    return render_template('index.html')


# predict and display it on result.html page
@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['upload-button']
    image = Image.open(image)
    image = image.resize((180, 180))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    result = class_names[np.argmax(score)]
    percentage = 100 * np.max(score)
    return render_template('results.html', result=result, percentage=percentage)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
