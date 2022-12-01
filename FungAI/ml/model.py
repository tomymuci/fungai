import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import applications


def initialize_model(metrics = ['accuracy'], loss = 'categorical_crossentropy', learning_rate = 0.001):
    '''Initialize a model'''

    base_model = applications.efficientnet.EfficientNetB3(include_top = False, weights = "imagenet", input_shape = (100, 100, 3), pooling = 'max')

    base_model.trainable = True
    x = base_model.output

    x = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001 )(x)
    x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                    bias_regularizer = regularizers.l1(0.006), activation = 'relu')(x)
    x = Dropout(rate = .4, seed=123)(x)
    output = Dense(9, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(Adamax(learning_rate = learning_rate), loss = loss, metrics = metrics)

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
