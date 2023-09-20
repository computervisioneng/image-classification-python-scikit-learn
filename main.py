# https://www.microsoft.com/en-US/download/details.aspx?id=54765
# https://www.kaggle.com/competitions/dogs-vs-cats/data
# this script is examined for a total dataset lower than 3000 images due to memory limitations and CPU.
# The highest performance is feasible with GPU on googlecolab.
# This script successfully manges image corruptions in the dataset

import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# prepare data
input_dir = "./PetImages"
categories = ['Cat', 'Dog']
data = []
labels = []


for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        try:
            img = imread(img_path)
            img = resize(img, (20, 20))
            flattened_img = img.flatten()
            data.append(flattened_img)
            labels.append(category_idx)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            print("reading next image")
            continue  # Skip this image and continue with the next one

data = np.asarray(data)
labels = np.asarray(labels)

print("data and labels are ready for training")
# train/split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                    shuffle=True, stratify=labels)
print("splitting dataset is completed")

# train classifier
classifier = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameters)
print("start training")
grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)
print('{}% of samples were correctly classified'.format(str(score * 100)))
pickle.dump(best_estimator, open('./model.p', 'wb'))
