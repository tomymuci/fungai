import os
import pickle
import shutil

def save_model(model) :

    if model is not None:
        if "models" not in os.listdir(".") :
            os.mkdir("models")
        else :
            shutil.rmtree("models")
            os.mkdir("models")
        pickle.dump(model, open('models/model.pkl', 'wb'))
    else :
        print("\n 🍄 model is None, cannot save")

    pass


def load_model() :

    if "models" not in os.listdir(".") :
        return None

    model = pickle.load(open('models/model.pkl', 'rb'))

    return model
