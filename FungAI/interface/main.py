import numpy as np
import os
import shutil
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from FungAI.data_sources.load import load_local
from FungAI.ml.model import initialize_model, train_model, evaluate_model
from FungAI.ml.registry import save_model, load_model

def preprocessor() :
    '''Load the data (from local for now), preprocess it and save it'''

    if "processed" not in os.listdir(".") :
        os.mkdir("processed")
    else :
        shutil.rmtree("processed")
        os.mkdir("processed")

    print("\n 🍄 Loading images...\n")

    _X, _y = load_local()

    print("\n 🍄 Loading done, preprocessing starting...\n")

    labels = np.unique(_y)
    encoder = LabelBinarizer()
    encoder.fit(labels)

    y = encoder.transform(_y)
    X = _X / 255

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    print("\n 🍄 Processing done, saving...\n")

    np.save("processed/X_train.npy", X_train)
    np.save("processed/y_train.npy", y_train)
    np.save("processed/X_test.npy", X_test)
    np.save("processed/y_test.npy", y_test)

    print("\n 🍄 Data saved...\n")

    return None

def train() :

    print("\n 🍄 Initializing model...\n")

    if "processed" not in os.listdir(".") :
        print("\n 🍄 no saved data, run preprocessing first.\n")
        return None

    X_train = np.load("processed/X_train.npy")
    y_train = np.load("processed/y_train.npy")

    model = initialize_model()

    print("\n 🍄 Training model...\n")

    model, history = train_model(model = model, X = X_train, y = y_train)

    print("\n 🍄 Model trained\n")

    print(f"\n val_accuracy : {history.history['val_accuracy']}")

    print("\n 🍄 Saving model...\n")

    save_model(model)

    print("\n 🍄 Model saved\n")

    return model, history

def evaluate() :

    print("\n 🍄 Loading model and data...\n")

    model = load_model()

    if model is None :
        print("\n 🍄 There is no saved model, run training first.\n")
        return None

    X_test = np.load("processed/X_test.npy")
    y_test = np.load("processed/y_test.npy")

    print("\n 🍄 Evaluating model...\n")

    metrics = evaluate_model(model, X = X_test, y = y_test)

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"\n🍄 model evaluated: loss {round(loss, 2)} accuracy {round(accuracy, 2)}")

    return None

def pred() :
    pass
