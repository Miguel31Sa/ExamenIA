import os
import numpy as np
import tensorflow as tf  
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import base64
import cv2
from io import BytesIO

# Configuración de la aplicación
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model("modelo_caridad.h5")

# Categorías de predicción
CATEGORIES = ['Extremo', 'Intermedio', 'Inicial']  

# Preparacion de la camara
def preprocess_image_from_camera(image):
    image = Image.open(BytesIO(base64.b64decode(image.split(',')[1])))  
    image = np.array(image.resize((224, 224)))  
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)
    return image

# Preprocesar imagen cargada desde el formulario
def preprocess_image_from_file(file_path):
    image = cv2.imread(file_path) 
    image = cv2.resize(image, (224, 224))  
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'file' in request.files:  # Imagen cargada desde el formulario
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', prediction="No se seleccionó ninguna imagen.")
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_preprocessed = preprocess_image_from_file(filepath)
            prediction = model.predict(image_preprocessed)
            predicted_class = CATEGORIES[np.argmax(prediction)]
        elif request.is_json:  
            data = request.get_json()
            image_data = data.get('image')
            image_preprocessed = preprocess_image_from_camera(image_data)
            prediction = model.predict(image_preprocessed)
            predicted_class = CATEGORIES[np.argmax(prediction)]
        
        return render_template('index.html', prediction=predicted_class)
    
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
