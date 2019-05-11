from keras.preprocessing import image
import numpy as np
import pickle
import cv2

def cnn(kp):
    classifier = pickle.load(open("data/model_saved.sav", 'rb'))

    inp = np.expand_dims(kp, axis = 0)
    result = classifier.predict_classes(inp)

    if result[0][0] <=0.5:
        pred = "Recolored"
    else:
        pred = "Original"

    return pred
