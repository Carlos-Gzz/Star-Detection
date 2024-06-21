import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

def load_data(file):
    data = np.load(file)
    images = data['images']
    labels = data['labels']
    return images, labels

def main():
    model = load_model('models/star_detection_model.keras')
    images, labels = load_data('data/preprocessed_data.npz')
    
    predictions = (model.predict(images) > 0.5).astype("int32")
    print(classification_report(labels, predictions, target_names=['Non-Star', 'Star']))

if __name__ == '__main__':
    main()
