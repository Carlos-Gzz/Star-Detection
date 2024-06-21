import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

def load_your_model():
    # Load the trained model
    model = load_model('models/star_detection_model.keras')
    # Recompile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_frame(frame):
    # Resize the frame to the size expected by the model
    frame = cv2.resize(frame, (256, 256))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def main():
    model = load_your_model()

    # Open a connection to the webcam (Razer camera - my default camera on MS)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        prediction = model.predict(processed_frame)
        print(f'Prediction: {prediction}')  # Print the prediction value

        # Display the result on the frame
        if prediction > 0.5:
            cv2.putText(frame, 'Five-Pointed Star Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'No Star Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Star Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
