
# predict.py
# Loads trained model if TensorFlow is available; otherwise uses a simple heuristic fallback.
import os
import numpy as np
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'pest_detector.h5')

def heuristic_predict(img_path):
    # Simple heuristic: compute average green vs red to guess healthy vs diseased (demo only)
    img = Image.open(img_path).convert('RGB').resize((128,128))
    arr = np.array(img).astype(float)
    avg_r = arr[:,:,0].mean()
    avg_g = arr[:,:,1].mean()
    # If green is significantly higher than red -> healthy else diseased
    if avg_g - avg_r > 10:
        return 'healthy', 0.75
    else:
        return 'diseased', 0.65

def predict_image(img_path):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        from tensorflow.keras.preprocessing import image
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        class_index = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))
        class_names = sorted([d for d in os.listdir(os.path.join(os.path.dirname(__file__), '..', 'data')) if os.path.isdir(os.path.join(os.path.dirname(__file__), '..', 'data', d))])
        predicted_label = class_names[class_index] if class_index < len(class_names) else str(class_index)
        return predicted_label, confidence
    except Exception:
        # Fallback
        return heuristic_predict(img_path)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        label, conf = predict_image(path)
        print(f'Predicted: {label} ({conf*100:.2f}%)')
    else:
        print('Usage: python predict.py <image_path>')
