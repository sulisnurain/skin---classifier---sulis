from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("model/model_kulit_final.h5")

classes = ['acne', 'dry', 'normal', 'oily']

rekomendasi = {

    "acne": [
        ("Facial Wash Acne", "Membersihkan wajah dan membantu mengurangi jerawat."),
        ("Gel Moisturizer", "Menjaga kelembapan tanpa menyumbat pori-pori."),
        ("Sunscreen", "Melindungi kulit dari paparan sinar UV.")
    ],

    "dry": [
        ("Hydrating Cleanser", "Membersihkan kulit tanpa membuatnya semakin kering."),
        ("Moisturizer", "Menjaga kulit tetap lembap dan halus."),
        ("Sunscreen", "Melindungi kulit dari kerusakan akibat sinar matahari.")
    ],

    "normal": [
        ("Gentle Cleanser", "Membersihkan wajah dengan lembut."),
        ("Moisturizer", "Menjaga keseimbangan kelembapan kulit."),
        ("Sunscreen", "Melindungi kulit dari sinar UV.")
    ],

    "oily": [
        ("Oil Control Cleanser", "Mengurangi minyak berlebih pada wajah."),
        ("Gel Moisturizer", "Memberikan hidrasi tanpa terasa berat."),
        ("Sunscreen", "Melindungi kulit dari sinar matahari.")
    ]
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']

    filepath = os.path.join("static/uploads", file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224,224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    hasil = classes[np.argmax(pred)]

    confidence = round(np.max(pred) * 100, 2)

    skincare = rekomendasi[hasil]

    return render_template(
        'result.html',
        gambar=filepath,
        hasil=hasil,
        confidence=confidence,
        skincare=skincare
    )
if __name__ == "__main__":
    app.run(debug=True)