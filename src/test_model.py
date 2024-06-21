import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_your_model():
    model = load_model('models/star_detection_model.keras')
    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def main():
    model = load_your_model()

    # Path to a sample image from your dataset
    sample_image_path = 'C:/Users/carlo/OneDrive/Documents/Developer/Python/Star-Detection/data/processed/non_star_0.png'

    processed_image = preprocess_image(sample_image_path)
    prediction = model.predict(processed_image)
    print(f'Prediction for sample image: {prediction}')

if __name__ == '__main__':
    main()
