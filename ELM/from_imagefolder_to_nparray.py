# two classes (class1, class2)
# only replace with a directory of yours
# finally .npy files saved in your directory with names train.npy, #test.npy, train_labels.npy, test_labels.npy
import cv2
import glob
import numpy as np
def load_train_val_data():
    #Train data
    train = []
    train_labels = []
    files = glob.glob ("jaffe2/train/POS/*.tiff") # your image path
    for myFile in files:
        image = cv2.imread (myFile,0)
        train.append (image)
        train_labels.append(1)
        files = glob.glob ("jaffe2/train/NEG/*.tiff")
    for myFile in files:
        image = cv2.imread (myFile,0)
        train.append (image)
        train_labels.append(0)
    train = np.array(train,dtype='float32') #as mnist
    train_labels = np.array(train_labels,dtype='int') #as mnist
# convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
# for example (120 * 40 * 40 * 3)-> (120 * 4800)
    train = np.reshape(train,[train.shape[0],train.shape[1]*train.shape[2]])

# save numpy array as .npy formats
    np.save('train',train)
    np.save('train_labels',train_labels)

#Test data
    test = []
    test_labels = []
    files = glob.glob ("jaffe2/validation/POS/*.tiff")
    for myFile in files:
        image = cv2.imread (myFile,0)
        test.append (image)
        test_labels.append(1) # class1
    files = glob.glob ("jaffe2/validation/NEG/*.tiff")
    for myFile in files:
        image = cv2.imread (myFile,0)
        test.append (image)
        test_labels.append(0) # class2

    test = np.array(test,dtype='float32') #as mnist example
    test_labels = np.array(test_labels,dtype='int') #as mnist
    test = np.reshape(test,[test.shape[0],test.shape[1]*test.shape[2]])
    return (train, train_labels,test,test_labels)
