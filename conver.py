import tensorflow as tf

# Load the model without compiling (no need for loss/functions)
model = tf.keras.models.load_model("BFP02D2x4s40N2T035.h5", compile=False)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save it
with open("BFP02D2x4s40N2T035.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversion complete!")
