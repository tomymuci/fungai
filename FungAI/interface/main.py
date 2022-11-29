import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


from FungAI.data_sources.load import load_local, load_cloud
from FungAI.ml.model import initialize_model, train_model

def preprocessor() :
    '''Load the data (from local for now) and preprocess it'''

    _X, _y = load_local()

    labels = np.unique(_y)
    encoder = LabelBinarizer()
    encoder.fit(labels)

    y = encoder.transform(_y)
    X = _X / 255

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    np.save("processed/X.npy", X)
    np.save("processed/y.npy", y)

    return None

def train() :

    X = np.load("processed/X.npy")
    y = np.load("processed/y.npy")

    model = initialize_model()

    model, history = train_model(model = model, X = X, y = y)

    print(f"\n val_accuracy : {history.history['val_accuracy']}")

    return model, history

def evaluate() :
    pass

def pred() :
    pass
