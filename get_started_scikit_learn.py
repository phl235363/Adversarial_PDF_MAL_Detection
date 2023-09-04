"""
The script demonstrates a simple example of using ART with scikit-learn. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
from sklearn.svm import SVC
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier
from art.utils import load_mnist

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# Step 1: Load the MNIST dataset

# (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 1a: Flatten dataset

# nb_samples_train = x_train.shape[0]
# nb_samples_test = x_test.shape[0]
# x_train = x_train.reshape((nb_samples_train, 28 * 28))
# x_test = x_test.reshape((nb_samples_test, 28 * 28))

X = np.load("X_clean.csv.npy")
y = np.load("y_clean.csv.npy")
# X = np.loadtxt("X_clean.csv", delimiter=",", skiprows=1)
# y = np.loadtxt("y_clean.csv", delimiter=",", skiprows=1)
# X = np.delete(X, [0], axis=1)
# y = np.delete(y, [0], axis=1)
X = X.astype('float')
y = y.astype('int') 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create an instance of the OneHotEncoder
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

encoder = OneHotEncoder()
# Fit and transform the data
y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()

encoder = OneHotEncoder()
# Fit and transform the data
y_test = encoder.fit_transform(y_test.reshape(-1, 1)).toarray()

# Step 2: Create the model
model = SVC(C=0.10, kernel="rbf")

# Step 3: Create the ART classifier

classifier = SklearnClassifier(model=model, clip_values=(0, 1))

# Step 4: Train the ART classifier

classifier.fit(X_train_scaled, y_train)

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(X_test_scaled)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.8)
x_test_adv = attack.generate(x=X_test_scaled)

# print(X_test_scaled)
# print(x_test_adv)

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
