import pandas as pd
import numpy as np
import os
import shutil
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from random import randint
from PIL import Image

from FungAI.data_sources.load import make_dataframes, make_gens
from FungAI.ml.model import make_model, train_model, evaluate_model, predict_new
from FungAI.ml.registry import save_model_local, load_model_local, save_model_mlflow, load_model_mlflow

from FungAI.ml.params import LOCAL_DATA_PROCESSED_PATH, DATA_SOURCE, DATA_SAVE, DATA_LOAD, MODEL_SAVE, MODEL_LOAD, LOCAL_DATA_PATH




def preprocessor() :
    '''Load the data (from local for now), preprocess it and save it.'''


    print("\n 🍄 Loading images...\n")

    if DATA_SOURCE == "local" :

        if LOCAL_DATA_PROCESSED_PATH not in os.listdir(".") :
            os.mkdir(LOCAL_DATA_PROCESSED_PATH)
        else :
            shutil.rmtree(LOCAL_DATA_PROCESSED_PATH)
            os.mkdir(LOCAL_DATA_PROCESSED_PATH)

        train_df, test_df, valid_df = make_dataframes(LOCAL_DATA_PATH) #Loading the local df's

        #This already brings the data set splitted into 3

    elif DATA_SOURCE == 'cloud' :

        print("\n❗️Data not loaded❗️\n 🍄 Only local source available for the moment.\n")
        return None

    print("\n 🍄 Loading done, preprocessing starting...\n")


    #train_df = trim(train_df, max_samples=240, column='labels')

    print("\n 🍄 Processing done, saving...\n")

    if DATA_SAVE == "local" :

        train_df.to_csv(f"{LOCAL_DATA_PROCESSED_PATH}/train_df.csv")
        test_df.to_csv(f"{LOCAL_DATA_PROCESSED_PATH}/test_df.csv")
        valid_df.to_csv(f"{LOCAL_DATA_PROCESSED_PATH}/valid_df.csv")


    elif DATA_SAVE == 'cloud' :

        print("\n❗️Data not saved❗️\n 🍄 Only local saving available for the moment.\n")
        return None

    print("\n 🍄 Data saved\n")


    return train_df, test_df, valid_df

def train() :
    '''Train a model with the saved data. PLEASE NOTe THAT THE KERAS DATAFRAME ITERATOR IS BEING DONE HERE!!!!'''


    params = {'epochs' : 10,
              'batch_size' : 30,
              'patience' : 5,
              'metrics' : ['accuracy', 'AUC', 'precision'],
              'loss' : 'categorical_crossentropy',
              'learning_rate' : 0.001
             }

    print("\n 🍄 Getting data...\n")

    if DATA_LOAD == "local" :

        if "train_df.csv" not in os.listdir(LOCAL_DATA_PROCESSED_PATH) :
            print("\n❗️There is no saved data❗️\n 🍄 Run preprocessing first.\n")
            return None

        train_df = pd.read_csv(f"{LOCAL_DATA_PROCESSED_PATH}/train_df.csv")
        test_df = pd.read_csv(f"{LOCAL_DATA_PROCESSED_PATH}/test_df.csv")
        valid_df = pd.read_csv(f"{LOCAL_DATA_PROCESSED_PATH}/valid_df.csv")



    elif DATA_LOAD == "cloud" :
        print("\n❗️Data not loaded❗️\n 🍄 Only local source available for the moment.\n")
        return None

    print("\n 🍄 Making Iterators...\n")

    train_gen, test_gen, valid_gen = make_gens(batch_size= params['batch_size'], ycol='labels', train_df=train_df, test_df=test_df, valid_df=valid_df, img_size=(200,280))

    print("\n 🍄 Initializing model...\n")

    model = make_model()

    print("\n 🍄 Training model...\n")

    model, history = train_model(model = model, train_gen=train_gen,
                                 valid_gen=valid_gen, epochs=params['epochs'], patience=params['patience'],
                                 shuffle=False)

    print("\n 🍄 Model trained\n")


    print("\n 🍄 Saving model...\n")

    if MODEL_SAVE == 'local' :
        message = save_model_local(model)
    elif MODEL_SAVE == "cloud" :
        message = save_model_mlflow(model, params)

    else :
        print("\n❗️Model not saved❗️\n")
        return None

    print(message)

    return model, history, test_gen

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

    #####---TEST---#####

    new_image = Image.open("raw_data/Mushrooms/Amanita/254_Lyi3XcTltRs.jpg")


    #####---END---#####

    # if new_image is None :
    #     print("\ngot no images, will load a random local image for tests\n")
    #     rdm_nb = randint(0, 3)


    #     if rdm_nb == 0 :
    #         new_image = Image.open("FungAI/Agaricus_campestre.jpeg")
    #         type = "Agaricus Campestre"
    #     elif rdm_nb == 1 :
    #         new_image = Image.open("FungAI/amanita_muscaria.jpeg")
    #         type = "Amanita Muscaria"
    #     elif rdm_nb == 2 :
    #         new_image = Image.open("FungAI/boletus_edulis.jpeg")
    #         type = "Boletus Edulis"
    # else:
    #     new_image = Image.open(new_image)


    print("\n 🍄 Processing image...\n")

    trans_img = np.array(new_image.resize((200, 280)))
    X = np.concatenate(trans_img, axis = 0).reshape((1, 200, 280, 3))

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
        print(f"{label} : {round(pred, 2)*100} %")

    if new_image is None :
        print(f"\n 🍄 It is suppose to be {type} 🙃\n")

    return genus
