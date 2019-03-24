#from keras.datasets import mnist
from keras.utils import to_categorical
from elm2 import ELM
import argparse
import os
import numpy as np
from from_imagefolder_to_nparray_fer import load_train_val_data


parser = argparse.ArgumentParser()
parser.add_argument('--n_hidden_nodes', type=int, default=1000)
parser.add_argument('--loss',
    choices=['mean_squared_error', 'mean_absolute_error'],
    default='mean_squared_error',
)
parser.add_argument('--activation',
    choices=['sigmoid', 'identity'],
    default='sigmoid',
)

def softmax(x):
    c = np.max(x, axis=-1)
    upper = np.exp(x - c)
    lower = np.sum(upper, axis=-1)
    return upper / lower

def main(args):
    # ===============================
    # Load dataset x and labelset t
    # ===============================
    n_classes = 2
    (x_train, t_train, x_test, t_test) = load_train_val_data()
    # ===============================
    # Preprocess
    # ===============================
    x_train = x_train.astype(np.float32) / 255.
    x_train = x_train.reshape(-1, 48**2)
    x_test = x_test.astype(np.float32) / 255.
    x_test = x_test.reshape(-1, 48**2)
    t_train = to_categorical(t_train, n_classes).astype(np.float32)
    t_test = to_categorical(t_test, n_classes).astype(np.float32)

    # ===============================
    # Instantiate ELM
    # ===============================
    model = ELM(
        n_input_nodes=48**2,
        n_hidden_nodes=args.n_hidden_nodes,
        n_output_nodes=n_classes,
        loss=args.loss,
        activation=args.activation,
        name='elm',
    )

    # ===============================
    # Training
    # ===============================
    model.fit(x_train, t_train)
    train_loss, train_acc = model.evaluate(x_train, t_train, metrics=['loss', 'accuracy'])
    print('train_loss: %f' % train_loss)
    print('train_acc: %f' % train_acc)

    # ===============================
    # Validation
    # ===============================
    val_loss, val_acc = model.evaluate(x_test, t_test, metrics=['loss', 'accuracy'])
    print('val_loss: %f' % val_loss)
    print('val_acc: %f' % val_acc)

    # ===============================
    # Prediction
    # ===============================
    print(x_test[:2])
    x = x_test[:2]
    t = t_test[:2]
    y = softmax(model.predict(x))

    for i in range(len(y)):
        print('---------- prediction %d ----------' % (i+1))
        class_pred = np.argmax(y[i])
        prob_pred = y[i][class_pred]
        class_true = np.argmax(t[i])
        print('prediction:')
        print('\tclass: %d, probability: %f' % (class_pred, prob_pred))
        print('\tclass (true): %d' % class_true)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
