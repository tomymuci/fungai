import numpy as np
import os
import shutil
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from random import randint
from PIL import Image

from FungAI.data_sources.load import make_dataframes, trim, make_gens
from FungAI.ml.model import make_model, train_model, evaluate_model, predict_new, F1_score
from FungAI.ml.registry import save_model_local, load_model_local, save_model_mlflow, load_model_mlflow

from FungAI.ml.params import LOCAL_DATA_PROCESSED_PATH, DATA_SOURCE, DATA_SAVE, DATA_LOAD, MODEL_SAVE, MODEL_LOAD




def preprocessor() :
    '''Load the data (from local for now), preprocess it and save it.'''


    print("\n 🍄 Loading images...\n")

    if DATA_SOURCE == "local" :

        if LOCAL_DATA_PROCESSED_PATH not in os.listdir(".") :
            os.mkdir(LOCAL_DATA_PROCESSED_PATH)
        else :
            shutil.rmtree(LOCAL_DATA_PROCESSED_PATH)
            os.mkdir(LOCAL_DATA_PROCESSED_PATH)

        train_df, test_df, valid_df = make_dataframes(train_dir,test_dir, val_dir) #Loading the local df's
        #This already brings the data set splitted into 3

    elif DATA_SOURCE == 'cloud' :

        print("\n❗️Data not loaded❗️\n 🍄 Only local source available for the moment.\n")
        return None

    print("\n 🍄 Loading done, preprocessing starting...\n")

    max_samples=240
    min_samples=240
    column='labels'

    train_df = trim(train_df, max_samples, min_samples, column)


    img_size=(200,280)# reduce image size to reduce training time but is a trade off because it may reduce model performance
    batch_size=30
    ycol='labels'


    train_gen, test_gen, valid_gen, test_steps= make_gens(batch_size, ycol, train_df, test_df, valid_df, img_size)

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

    return train_gen, test_gen, valid_gen

def train() :
    '''Train a model with the saved data.'''

    params = {'epochs' : 1,
              'batch_size' : 16,
<<<<<<< HEAD
              'patience' : 10,
              'metrics' : ['accuracy',F1_score, 'AUC'],
=======
              'patience' : 5,
              'metrics' : ['accuracy'],
>>>>>>> 5397d2b7bfb5b289a6392cc520611edcf2fecbf1
              'loss' : 'categorical_crossentropy',
              'learning_rate' : 0.001
             }

    print("\n 🍄 Getting data...\n")

    if DATA_LOAD == "local" :

        if "X_train.npy" not in os.listdir(LOCAL_DATA_PROCESSED_PATH) :
            print("\n❗️There is no saved data❗️\n 🍄 Run preprocessing first.\n")
            return None

        X_train = np.load(f"{LOCAL_DATA_PROCESSED_PATH}/X_train.npy")
        y_train = np.load(f"{LOCAL_DATA_PROCESSED_PATH}/y_train.npy")

    elif DATA_LOAD == "cloud" :
        print("\n❗️Data not loaded❗️\n 🍄 Only local source available for the moment.\n")
        return None

    print("\n 🍄 Initializing model...\n")

    model = make_model()

    print("\n 🍄 Training model...\n")

    model, history = train_model(model = model,
                                 valid_gen=valid_gen)

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

    print("\n 🍄 Loading model...\n")

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

    print("\n 🍄 Loading data...\n")

    if DATA_LOAD == 'local' :
        X_test = np.load(f"{LOCAL_DATA_PROCESSED_PATH}/X_test.npy")
        y_test = np.load(f"{LOCAL_DATA_PROCESSED_PATH}/y_test.npy")
    elif DATA_LOAD == 'cloud' :
        print("\n❗️Data not loaded❗️\n 🍄 Only local source available for the moment.\n")
        return None

    print("\n 🍄 Evaluating model...\n")

    metrics = evaluate_model(model, X = X_test, y = y_test)

    loss = metrics["loss"]
    accuracy = metrics["Accuracy"]

    print(f"\n 🍄 Model evaluated : loss {round(loss, 2)} accuracy {round(accuracy, 2)}")

    return metrics

def pred(new_image = None) :

    print("\n 🍄 Loading image...\n")

    if new_image is None :
        print("\ngot no images, will load a random local image for tests\n")
        rdm_nb = randint(0, 3)

        if rdm_nb == 0 :
            new_image = Image.open("FungAI/Agaricus_campestre.jpeg")
            type = "Agaricus Campestre"
        elif rdm_nb == 1 :
            new_image = Image.open("FungAI/amanita_muscaria.jpeg")
            type = "Amanita Muscaria"
        elif rdm_nb == 2 :
            new_image = Image.open("FungAI/boletus_edulis.jpeg")
            type = "Boletus Edulis"


    print("\n 🍄 Processing image...\n")

    trans_img = np.array(new_image.resize((100, 100)))
    X = np.concatenate(trans_img, axis = 0).reshape((1, 100, 100, 3))

    print("\n 🍄 Loading Model...\n")

    if MODEL_LOAD == "local" :
        model = load_model_local()
    elif MODEL_LOAD == 'cloud' :
        model = load_model_mlflow()
    else :
        print("\n❗️Model not loaded❗️\n")
        return None

    print("\n 🍄 Making a prediction...\n")

    prediction = predict_new(model = model, X = X)[0]
    labels = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']

    print("\n 🍄 Got a prediction!\n")

    genus = {}
    for label, pred in zip(labels, prediction) :
        genus[label] = f"{pred}"
        print(f"{label} : {pred*100} %")

    if new_image is None :
        print(f"\n 🍄 It is suppose to be {type} 🙃\n")

    return genus
