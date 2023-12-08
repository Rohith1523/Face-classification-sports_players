import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import classification_report



# Function to create and compile the model
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Function to load data and train the model
def train_model(main_folder, num_classes):
    # Initialize variables for image data
    batch_size = 128
    img_height, img_width = 150, 150  # Adjust as needed
    input_shape = (img_height, img_width, 3)

    train_data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

    train_generator = train_data_gen.flow_from_directory(
        main_folder,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_data_gen.flow_from_directory(
        main_folder,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    model = create_model(input_shape, num_classes)

    # Train the model
    epochs = 500  # Adjust as needed

    steps_per_epoch = train_generator.samples // batch_size if train_generator.samples >= batch_size else 1
    validation_steps = validation_generator.samples // batch_size if validation_generator.samples >= batch_size else 1
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps
    )

    return model

# Ask for paths and start training
num_classes = 5  # Change based on the number of folders (persons)

main_folder = r'D:\STUDY\S3\DEEP LEARNING\image classification\rj\cropped'# Replace with the path to your dataset folder
# Training the model
model = train_model(main_folder, num_classes)

model.save('model.keras')

print("--------------------------------------\n")
print("Model Prediction.\n")
root_dir = r"D:\STUDY\S3\DEEP LEARNING\image classification\rj\cropped"
celebrities=os.listdir(root_dir)

def make_prediction(img, model, celebrities):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = Image.fromarray(img)
    img = img.resize((150, 150))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    input_img = tf.keras.utils.normalize(input_img, axis=1) 
    predictions = model.predict(input_img)
    predicted_class = np.argmax(predictions)
    celebrity_name = celebrities[predicted_class]
    print(f"Predicted Celebrity: {celebrity_name}")

        
make_prediction(os.path.join(root_dir, "lionel_messi", "lionel_messi6.png"), model, celebrities)
make_prediction(os.path.join(root_dir, "roger_federer", "roger_federer4.png"), model, celebrities)
make_prediction(os.path.join(root_dir, "virat_kohli", "virat_kohli6.png"), model, celebrities)
make_prediction(os.path.join(root_dir, "maria_sharapova", "maria_sharapova4.png"), model, celebrities)
make_prediction(os.path.join(root_dir, "serena_williams", "serena_williams7.png"), model, celebrities)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig(r'D:\STUDY\S3\DEEP LEARNING\image classification\rj\cropped\accuracy_plot.png')

# Clear the previous plot
plt.clf()

# Plot and save loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(r'D:\STUDY\S3\DEEP LEARNING\image classification\rj\cropped\loss_plot.png')