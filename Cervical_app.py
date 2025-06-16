import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import gdown

# ---- Step 1: Download model from Google Drive if not present ----
model_path = 'EfficientNet-cervical.keras'
if not os.path.exists(model_path):
    file_id = '1TupwkSxykApCVHyZp-geuXlzncbO_LT_'
    gdown.download(f'https://drive.google.com/uc?id={file_id}', model_path, quiet=False)

# ---- Step 2: Initialize Flask app ----
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ---- Step 3: Load trained model ----
model = load_model(model_path)

# ---- Step 4: Define class labels ----
class_names = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']

# ---- Step 5: Routes ----

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Upload and predict
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Predict
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_index]
            confidence = float(np.max(predictions[0]) * 100)

            return render_template('result.html',
                                   filename=filename,
                                   predicted_class=predicted_class,
                                   confidence=round(confidence, 2))

        return redirect(request.url)

    return render_template('upload.html')

# Serve uploaded image
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# ---- Step 6: Run locally ----
if __name__ == '__main__':
    app.run(debug=True)
