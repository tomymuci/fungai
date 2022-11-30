import numpy as np
import os
import shutil
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from FungAI.data_sources.load import load_local
from FungAI.ml.model import initialize_model, train_model, evaluate_model
from FungAI.ml.registry import save_model_local, load_model_local, save_model_mlflow, load_model_mlflow

from FungAI.ml.params import LOCAL_DATA_PROCESSED_PATH, DATA_SOURCE, DATA_SAVE, DATA_LOAD, MODEL_SAVE, MODEL_LOAD

def preprocessor(prediction = None) :
    '''Load the data (from local for now), preprocess it and save it.'''


    print("\n 🍄 Loading images...\n")

    if DATA_SOURCE == "local" :

        if LOCAL_DATA_PROCESSED_PATH not in os.listdir(".") :
            os.mkdir(LOCAL_DATA_PROCESSED_PATH)
        else :
            shutil.rmtree(LOCAL_DATA_PROCESSED_PATH)
            os.mkdir(LOCAL_DATA_PROCESSED_PATH)

        _X, _y = load_local()

    elif DATA_SOURCE == 'cloud' :

        print("\n❗️Data not loaded❗️\n 🍄 Only local source available for the moment.\n")
        return None

    print("\n 🍄 Loading done, preprocessing starting...\n")

    labels = np.unique(_y)
    encoder = LabelBinarizer()
    encoder.fit(labels)

    y = encoder.transform(_y)
    X = _X / 255

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    print("\n 🍄 Processing done, saving...\n")

    if DATA_SAVE == "local" :

        np.save(f"{LOCAL_DATA_PROCESSED_PATH}/X_train.npy", X_train)
        np.save(f"{LOCAL_DATA_PROCESSED_PATH}/y_train.npy", y_train)
        np.save(f"{LOCAL_DATA_PROCESSED_PATH}/X_test.npy", X_test)
        np.save(f"{LOCAL_DATA_PROCESSED_PATH}/y_test.npy", y_test)

    elif DATA_SAVE == 'cloud' :

        print("\n❗️Data not saved❗️\n 🍄 Only local saving available for the moment.\n")
        return None

    print("\n 🍄 Data saved\n")

    return None

def train() :
    '''Train a model with the saved data.'''

    params = {'epochs' : 100,
              'batch_size' : 16,
              'patience' : 10,
              'metrics' : ['Accuracy'],
              'loss' : 'categorical_crossentropy'
             }

    print("\n 🍄 Initializing model...\n")

    if DATA_LOAD == "local" :

        if "X_train.npy" not in os.listdir(LOCAL_DATA_PROCESSED_PATH) :
            print("\n❗️There is no saved data❗️\n 🍄 Run preprocessing first.\n")
            return None

        X_train = np.load(f"{LOCAL_DATA_PROCESSED_PATH}/X_train.npy")
        y_train = np.load(f"{LOCAL_DATA_PROCESSED_PATH}/y_train.npy")

    elif DATA_LOAD == "cloud" :
        print("\n❗️Data not loaded❗️\n 🍄 Only local source available for the moment.\n")
        return None

    model = initialize_model(metrics = params['metrics'], loss = params["loss"])

    print("\n 🍄 Training model...\n")

    model, history = train_model(model = model,
                                 X = X_train,
                                 y = y_train,
                                 epochs = params["epochs"],
                                 batch_size = params["batch_size"],
                                 patience = params["patience"])

    print("\n 🍄 Model trained\n")

    metrics = {}

    for metric in params['metrics'] :
        metrics[metric] = np.max(history.history[f"val_{metric}"])
        print(f"\n val_{metric}: {metrics[metric]}")

    print("\n 🍄 Saving model...\n")

    if MODEL_SAVE == 'local' :
        message = save_model_local(model)
    elif MODEL_SAVE == "cloud" :
        message = save_model_mlflow(model, params, metrics)

    else :
        print("\n❗️Model not saved❗️\n")
        return None

    print(message)

    return model, history

def evaluate() :
    '''Evaluate a model with a saved model.'''

    print("\n 🍄 Loading model and data...\n")

    if MODEL_LOAD == "local" :
        model = load_model_local()
    elif MODEL_LOAD == 'cloud' :
        model = load_model_mlflow()
    else :
        print("\n❗️Model not loaded❗️\n")
        return None

    if model is None :
        print("\n❗️There is no saved model❗️\n 🍄 Run training first or change the loading parameters.\n")
        return None

    if DATA_LOAD == 'local' :
        X_test = np.load(f"{LOCAL_DATA_PROCESSED_PATH}/X_test.npy")
        y_test = np.load(f"{LOCAL_DATA_PROCESSED_PATH}/y_test.npy")
    elif DATA_LOAD == 'cloud' :
        print("\n❗️Data not loaded❗️\n 🍄 Only local source available for the moment.\n")
        return None

    print("\n 🍄 Evaluating model...\n")

    metrics = evaluate_model(model, X = X_test, y = y_test)

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"\n 🍄 Model evaluated : loss {round(loss, 2)} accuracy {round(accuracy, 2)}")

    return metrics

def pred(new_image) :
    pass
