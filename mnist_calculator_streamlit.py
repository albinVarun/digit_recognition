import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import io

class DigitRecognizer:
    def __init__(self):
        self.load_model()
        
    def load_model(self):
        # Create and compile model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        self.model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        
        # Train on MNIST dataset
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y_train = y_train
        self.model.fit(x_train, y_train, epochs=1, verbose=1)
    
    def predict_digit(self, image):
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Resize to 28x28
        img_array = cv2.resize(img_array, (28, 28))
        
        # Normalize and reshape
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Get prediction
        prediction = self.model.predict(img_array, verbose=0)
        digit = np.argmax(prediction[0])
        confidence = prediction[0][digit] * 100
        
        return digit, confidence, prediction[0]

def main():
    st.set_page_config(page_title="Handwritten Digit Calculator", layout="wide")
    st.title("Handwritten Digit Calculator")
    
    # Initialize the recognizer
    if 'recognizer' not in st.session_state:
        with st.spinner("Loading model... This might take a minute."):
            st.session_state.recognizer = DigitRecognizer()
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload a digit image")
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', width=280)
            
            # Add a recognize button
            if st.button("Recognize"):
                digit, confidence, probabilities = st.session_state.recognizer.predict_digit(image)
                
                with col2:
                    st.subheader("Prediction")
                    st.success(f"Predicted Digit: {digit}")
                    st.info(f"Confidence: {confidence:.2f}%")
                    
                    # Display probability distribution
                    st.write("Probability Distribution:")
                    prob_dict = {f"Digit {i}": float(probabilities[i]) * 100 for i in range(10)}
                    st.bar_chart(prob_dict)

if __name__ == "__main__":
    main()
