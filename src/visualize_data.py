import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_data(file):
    data = np.load(file)
    images = data['images']
    labels = data['labels']
    return images, labels

def visualize_data(images, labels, num_samples=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        ax = plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i])
        plt.title('Star' if labels[i] == 1 else 'Not Star')
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    images, labels = load_data('data/preprocessed_data.npz')
    visualize_data(images, labels, num_samples=10)
