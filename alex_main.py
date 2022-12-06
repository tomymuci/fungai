### Creating a replicate of the model py file but with the changes related to the Last model analysis
import pandas as pd
import numpy as np
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from alex_load import make_dataframes, trim, make_gens
from alex_model import make_model, f1_score, train_model
from tensorflow.keras.callbacks import EarlyStopping

from FungAI.data_sources.load import load_local
from FungAI.ml.model import initialize_model, train_model, evaluate_model, predict_new
from FungAI.ml.registry import save_model_local, load_model_local, save_model_mlflow, load_model_mlflow

from FungAI.ml.params import LOCAL_DATA_PROCESSED_PATH, DATA_SOURCE, DATA_SAVE, DATA_LOAD, MODEL_SAVE, MODEL_LOAD


def preprocessor():
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
              'patience' : 10,
              'metrics' : ['accuracy',f1_score, 'AUC'],
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







##########-----------SPLITTING TRAINING/VALIDATION/TEST SET-----------##########


## Now, we will proceed to split the data into validation


    def make_gens(batch_size=30, ycol='labels', train_df, test_df, valid_df, img_size=(200,280)):
        trgen=ImageDataGenerator(horizontal_flip=True)
        t_and_v_gen=ImageDataGenerator()
    #Create the training set based on the training_df created above
        train_gen=trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col=ycol, target_size=img_size,
                                        class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

    #Create the validation set based
        valid_gen=t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col=ycol, target_size=img_size,
                                        class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)

    #CREATING TEST SET

        # for the test_gen we want to calculate the batch size and test steps such that batch_size X test_steps= number of samples in test set
        # this insures that we go through all the sample in the test set exactly once.

        length=len(test_df)
        test_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80],reverse=True)[0]
        test_steps=int(length/test_batch_size)


        test_gen=t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col=ycol, target_size=img_size,
                                        class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)

        return train_gen, test_gen, valid_gen, test_steps


##########-----------SPLITTING TRAINING/VALIDATION/TEST SET-----------##########
