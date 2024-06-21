import os
import cv2
import numpy as np

def collect_images(star_dir, non_star_dir, output_dir, image_size=(256, 256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    star_images = [os.path.join(star_dir, f) for f in os.listdir(star_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    non_star_images = [os.path.join(non_star_dir, f) for f in os.listdir(non_star_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for i, img_path in enumerate(star_images):
        image = cv2.imread(img_path)
        image = cv2.resize(image, image_size)
        cv2.imwrite(os.path.join(output_dir, f'star_{i}.png'), image)

    for i, img_path in enumerate(non_star_images):
        image = cv2.imread(img_path)
        image = cv2.resize(image, image_size)
        cv2.imwrite(os.path.join(output_dir, f'non_star_{i}.png'), image)

    print(f'Collected {len(star_images)} star images and {len(non_star_images)} non-star images.')

def preprocess_images(input_dir, output_file):
    images = []
    labels = []

    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(os.path.join(input_dir, filename))
            image = cv2.resize(image, (256, 256))
            images.append(image)

            if filename.startswith('star_'):
                labels.append(1)
                print(f'Labeling {filename} as Star')
            elif filename.startswith('non_star_'):
                labels.append(0)
                print(f'Labeling {filename} as Non-Star')

            if len(images) % 50 == 0:
                print(f'Processed {len(images)} images.')

    images = np.array(images)
    labels = np.array(labels)

    np.savez(output_file, images=images, labels=labels)
    print(f'Preprocessed images saved to {output_file}')

if __name__ == '__main__':
    collect_images('data/raw/stars', 'data/raw/non_stars', 'data/processed')
    preprocess_images('data/processed', 'data/preprocessed_data.npz')
