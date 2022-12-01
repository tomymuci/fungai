import numpy as np

from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping


def initialize_model(metrics = ['accuracy'], loss = 'categorical_crossentropy'):
    '''Initialize a model'''

    model = models.Sequential()

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(32,(3,3), activation='relu',padding='same',input_shape=(100,100,3)))
    model.add(layers.Conv2D(32,(3,3), activation='relu',padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))

    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))

    ### Third Convolution & MaxPooling
    model.add(layers.Conv2D(16, (3,3), activation='relu', padding='same'))

    model.add(layers.Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))

    #Flattening the model
    model.add(layers.Flatten())

    #One more dropout layer with the respective dense layer to combine everything

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))


    ### Last layer - Classification Layer with 9 outputs corresponding to 9 mushrooms labels
    model.add(layers.Dense(9,activation='softmax'))

    ### Model compilation
    model.compile(loss = loss,
                  optimizer = 'adam',
                  metrics = metrics)

    return model

def train_model(model, X: np.ndarray, y: np.ndarray, epochs = 5, batch_size = 16, patience = 1, validation_split = 0.2) :
    '''Train a model.'''

    es = EarlyStopping(patience = patience, restore_best_weights = True)
    history = model.fit(X, y, batch_size = batch_size, epochs = epochs, callbacks = [es], verbose = 1, validation_split = validation_split)

    return model, history

def evaluate_model(model, X: np.ndarray, y: np.ndarray, batch_size = 16) :
    '''Evaluate a model.'''

    metrics = model.evaluate(x = X, y = y, batch_size = batch_size, verbose = 1, return_dict = True)

    return metrics

def predict_new(model, X, batch_size = 16) :
    '''Use the selected model to make a prediction.'''

    y_pred = model.predict(x = X, batch_size = batch_size, verbose = 1, callbacks = None)

    return y_pred
