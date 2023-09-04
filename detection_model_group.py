"""
The script demonstrates a simple example of using ART with scikit-learn. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
from sklearn.svm import SVC
import numpy as np
import random

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
X = X.astype("float")
y = y.astype("int")

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder()
# Fit and transform the data
y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()

encoder = OneHotEncoder()
# Fit and transform the data
y_test = encoder.fit_transform(y_test.reshape(-1, 1)).toarray()

# Step 2: Create the model
model = SVC(C=0.5, kernel="rbf")

# Step 3: Create the ART classifier
original_classifier = SklearnClassifier(model=model, clip_values=(0, 1))

# Step 4: Train the ART original_classifier
original_classifier.fit(X_train, y_train)

# Step 5: Evaluate the ART original_classifier on benign test examples
print("\n-----Adversarial attack demonstration-----")
original_predictions = original_classifier.predict(X_test)
original_accuracy = np.sum(
    np.argmax(original_predictions, axis=1) == np.argmax(y_test, axis=1)
) / len(y_test)
print("Accuracy on benign test examples: {}%".format(original_accuracy * 100))

# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=original_classifier, eps=0.8)
X_test_adv = attack.generate(x=X_test)

# Step 7: Evaluate the ART original_classifier on adversarial test examples
art_predictions = original_classifier.predict(X_test_adv)
art_accuracy = np.sum(
    np.argmax(art_predictions, axis=1) == np.argmax(y_test, axis=1)
) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(art_accuracy * 100))


# Generate detection model group
print("\n-----Detection model group generation-----")
model_group = []
kernels = ["linear", "poly", "rbf"]
for x in range(500):
    kernel = random.choice(kernels)
    C = random.uniform(0, 10)
    model = SVC(C=C, kernel=kernel)

    classifier = SklearnClassifier(model=model, clip_values=(0, 1))

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    accuracy = np.sum(
        np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)
    ) / len(y_test)
    # print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
    if accuracy >= original_accuracy:
        model_group.append(classifier)

print("Model group: {} models".format(len(model_group)))

# Detection Method
print("\n-----PIR calculation-----")


X_malicious = X[50:51]
X_base = X[X.shape[0] - 1: X.shape[0]]
y_malicious = np.array([[0.0, 1.0]])
y_base = np.array([[1.0, 0.0]])

attack = FastGradientMethod(estimator=original_classifier, eps=0.8)
X_malicious_adv = attack.generate(x=X_malicious)

base_PIR = 0
original_benign_prediction = original_classifier.predict(X_base)
for i in range(len(model_group)):
    model_group_benign_prediction = model_group[i].predict(X_base)
    if not (original_benign_prediction == model_group_benign_prediction).all():
        base_PIR = base_PIR + 1

print("Base PIR: {}".format(base_PIR))

PIR = 0
original_malicious_prediction = original_classifier.predict(X_malicious_adv)
for i in range(len(model_group)):
    model_group_malicious_prediction = model_group[i].predict(X_malicious_adv)
    if not (original_malicious_prediction == model_group_malicious_prediction).all():
        PIR = PIR + 1
print("PIR: {}".format(PIR))
if PIR > base_PIR:
    print("PIR > Base PIR - Adversarial Malicious examples!!!!!")
else:
    print("Benign :D")
