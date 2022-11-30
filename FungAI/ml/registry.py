import os
import pickle
import shutil
import mlflow

from FungAI.ml.params import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, MLFLOW_MODEL_NAME, LOCAL_REGISTRY_PATH, MLFLOW_MODEL_STATUS

def save_model_local(model = None) :
    '''Save a model in local directory'''

    if LOCAL_REGISTRY_PATH not in os.listdir(".") :
        os.mkdir(LOCAL_REGISTRY_PATH)

    if model is not None:
        shutil.rmtree(LOCAL_REGISTRY_PATH)
        os.mkdir(LOCAL_REGISTRY_PATH)
        pickle.dump(model, open(f'{LOCAL_REGISTRY_PATH}/model.pkl', 'wb'))
        message = "\n 🍄 Model saved\n"
    else :
        message = "\n❗️Model is None, cannot save❗️\n"

    return message


def load_model_local() :
    '''Load a model from local directory'''

    if LOCAL_REGISTRY_PATH not in os.listdir(".") :
        return None
    elif "model.pkl" not in os.listdir(LOCAL_REGISTRY_PATH) :
        return None


    model = pickle.load(open(f'{LOCAL_REGISTRY_PATH}/model.pkl', 'rb'))

    return model

def save_model_mlflow(model = None, params = None, metrics = None) :
    '''Save a model to the cloud'''

    if model is not None :
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name = MLFLOW_EXPERIMENT)

        with mlflow.start_run() :

            if params is not None :
                mlflow.log_params(params)

            if metrics is not None :
                mlflow.log_metrics(metrics)

            mlflow.keras.log_model(keras_model = model,
                                   artifact_path = "model",
                                   keras_module = "tensorflow.keras",
                                   registered_model_name = MLFLOW_MODEL_NAME)

        message = "\n 🍄 Model saved\n"
    else :
        message = "\n❗️Model is None, cannot save❗️\n"

    return message

def load_model_mlflow() :
    '''load a model from the cloud'''

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if MLFLOW_MODEL_STATUS == "developement" :
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/None"
    elif MLFLOW_MODEL_STATUS == "production" :
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/production"
    else :
        print("\n❗️Wrong MLflow status❗️\n")
        return None

    model = mlflow.keras.load_model(model_uri = model_uri)

    return model
