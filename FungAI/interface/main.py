import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


from FungAI.data_sources.load import load_local, load_cloud
from FungAI.ml.model import initialize_model, train_model, evaluate_model

def preprocessor() :
    '''Load the data (from local for now), preprocess it and save it'''

    if "processed" not in os.listdir(".") :
        os.mkdir("processed")
    else :
        os.remove("processed/X.npy")
        os.remove("processed/y.npy")

    print("\n 🍄 Loading images...\n")

    _X, _y = load_local()

    print("\n 🍄 Loading done, preprocessing starting...\n")

    labels = np.unique(_y)
    encoder = LabelBinarizer()
    encoder.fit(labels)

    y = encoder.transform(_y)
    X = _X / 255

    print("\n 🍄 Processing done, saving...\n")

    np.save("processed/X.npy", X)
    np.save("processed/y.npy", y)

    return None

def train() :

    print("\n 🍄 Initializing model...\n")

    X = np.load("processed/X.npy")
    y = np.load("processed/y.npy")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    model = initialize_model()

    print("\n 🍄 Training model...\n")

    model, history = train_model(model = model, X = X_train, y = y_train)

    print("\n 🍄 Model trained\n")

    print(f"\n val_accuracy : {history.history['val_accuracy']}")

    return model, history, X_test, y_test

def evaluate() :

    print("\n 🍄 Loading and training model...\n")

    model, history, X_test, y_test = train()

    print("\n 🍄 Evaluating model...\n")

    metrics = evaluate_model(model, X = X_test, y = y_test)

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"\n🍄 model evaluated: loss {round(loss, 2)} accuracy {round(accuracy, 2)}")

    return None

def pred() :
    pass
