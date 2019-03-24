# Standard scientific Python imports
import matplotlib.pyplot as plt
from from_imagefolder_to_nparray_fer import load_train_val_data
import numpy as np


# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics

    # ===============================
    # Load dataset x and labelset t
    # ===============================
n_classes = 2
(x_train, t_train, x_test, t_test) = load_train_val_data()

x_train = x_train.astype(np.float32) / 255.
x_train = x_train.reshape(-1, 48**2)
x_test = x_test.astype(np.float32) / 255.
x_test = x_test.reshape(-1, 48**2)
#t_train = to_categorical(t_train, n_classes).astype(np.float32)
#t_test = to_categorical(t_test, n_classes).astype(np.float32)

# Create a classifier: a support vector classifier
print(x_train.shape)
print(t_train.shape)
classifier = svm.SVC(gamma=0.0001)

# We learn the digits on the first half of the digits
classifier.fit(x_train, t_train)

# Now predict the value of the digit on the second half:
expected = t_test
predicted = classifier.predict(x_test)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
