"this is where we are suppose to load the parameters from the .env and that we'll call in others files."

import os

LOCAL_DATA_PATH = os.path.expanduser(os.environ.get("LOCAL_DATA_PATH"))
LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))
LOCAL_DATA_PROCESSED_PATH = os.path.expanduser(os.environ.get("LOCAL_DATA_PROCESSED_PATH"))

DATA_SOURCE = os.path.expanduser(os.environ.get("DATA_SOURCE"))
DATA_SAVE = os.path.expanduser(os.environ.get("DATA_SAVE"))
DATA_LOAD = os.path.expanduser(os.environ.get("DATA_LOAD"))
MODEL_SAVE = os.path.expanduser(os.environ.get("MODEL_SAVE"))
MODEL_LOAD = os.path.expanduser(os.environ.get("MODEL_LOAD"))

MLFLOW_TRACKING_URI = os.path.expanduser(os.environ.get("MLFLOW_TRACKING_URI"))
MLFLOW_EXPERIMENT = os.path.expanduser(os.environ.get("MLFLOW_EXPERIMENT"))
MLFLOW_MODEL_NAME = os.path.expanduser(os.environ.get("MLFLOW_MODEL_NAME"))
MLFLOW_MODEL_STATUS = os.path.expanduser(os.environ.get("MLFLOW_MODEL_STATUS"))


################## VALIDATIONS #################

env_valid_options = dict(DATA_SOURCE = ["local", "cloud"],
                         DATA_SAVE = ["local", "cloud"],
                         DATA_LOAD = ["local", "cloud"],
                         MODEL_SAVE = ["local", "cloud"],
                         MODEL_LOAD = ["local", "cloud"],
                         MLFLOW_MODEL_STATUS = ["developement", 'production']
                        )

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)
