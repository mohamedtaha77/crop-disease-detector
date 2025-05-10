from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
os.makedirs('static/uploads', exist_ok=True)


app = Flask(__name__)
model = load_model('model/model.h5')

IMG_SIZE = (128, 128)  # Change if your model used different size
CLASS_NAMES = [
    'Apple__Apple_scab', 'Apple__Black_rot', 'Apple__Cedar_apple_rust',
    'Apple__healthy', 'Corn__Common_rust', 'Potato__Early_blight',
    'Potato__healthy', 'Tomato__Early_blight', 'Tomato__healthy',
    'Tomato__Tomato_YellowLeafCurlVirus'  # adjust based on your classes
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    if request.method == 'POST':
        image = request.files['image']
        path = os.path.join('static/uploads', image.filename)
        image.save(path)

        img = load_img(path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        result = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(result)]
        prediction = predicted_class
        image_path = path

    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

