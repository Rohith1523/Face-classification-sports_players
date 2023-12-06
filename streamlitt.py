import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Title and description for the app
st.title('Image Classification OF Sports Players')
st.write('Upload an image for classification')

# Load the model
st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model.keras')  # Replace with your model path
    return model

model = load_model()

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    # Define class labels
class_labels = {
    0: "Lionel Messi",
    1: "Maria sharapova",
    2: "Roger federer",
    3: "Serena williams",
    4: "Virat kohli",
    }

# ...

if uploaded_file is not None:
    # ... (previous code remains the same)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    if predicted_class in class_labels:
        predicted_label = class_labels[predicted_class]
        st.title(f'Predicted Person: {predicted_label}')
    else:
        st.write(f'Predicted Person: Unknown')

