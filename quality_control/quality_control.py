import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model

def list_files_and_labels(main_directory, label):
    files = os.listdir(main_directory)
    file_paths = [os.path.join(main_directory, file) for file in files]
    labels = [label] * len(file_paths)
    return file_paths, labels

def blur_background(file_paths, blur_amount=15):
    image_data = cv2.imread(file_paths)
    gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    dilated_edges = cv2.dilate(edges, None, iterations=2)
    background = cv2.bitwise_not(dilated_edges)
    blurred_image = cv2.GaussianBlur(image_data, (blur_amount, blur_amount), 0)
    result = np.where(background[:, :, None].astype(bool), blurred_image, image_data)
    return result

# Read images and label them
def_front_files, def_front_labels = list_files_and_labels('/def_front/', 'def_front')
ok_front_files, ok_front_labels = list_files_and_labels('/ok_front/', 'ok_front')

# Combine all files and labels
all_files = def_front_files + ok_front_files
all_labels = def_front_labels + ok_front_labels

image_data_list = []
for file_path in all_files:
    image_data = cv2.imread(file_path)
    resized_image = cv2.resize(image_data, (64, 64))
    vectorized_image = resized_image.flatten()
    image_data_list.append(vectorized_image)

for i in range(len(all_files)):
    blur_background(all_files[i], blur_amount=15)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Extract feature vectors from images
X = np.array(image_data_list)
y = np.array(all_labels)

# Load VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor_layer = base_model.get_layer('block5_conv2')
feature_extractor = Model(inputs=base_model.input, outputs=feature_extractor_layer.output)

# Train MLP model using feature vectors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=300, solver="lbfgs")
clf.fit(X_train, y_train)

# Evaluate the model
predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy Score: ", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
