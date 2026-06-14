from tensorflow.keras.models import load_model

print("Mulai membaca model...")

model = load_model('model/model_kulit.h5')

print("Model berhasil dibuka!")
model.summary()