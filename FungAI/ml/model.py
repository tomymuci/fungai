import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import applications
from tensorflow.keras import backend as K

# defining F1_Score

def F1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val



##########-----------INITIALIZING MODEL-----------##########


def make_model(img_size=(200,280), lr=0.001, mod_num=3):
    img_shape=(img_size[0], img_size[1], 3)
    if mod_num == 3:
        base_model=tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max')
        msg='Created EfficientNet B3 model'
    else:
        base_model=tf.keras.applications.efficientnet.EfficientNetB7(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max')
        msg='Created EfficientNet B7 model'

    base_model.trainable=True


    x=base_model.output
    x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
    x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                    bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
    x=Dropout(rate=.4, seed=123)(x)
    output=Dense(class_count, activation='softmax')(x)

    model=Model(inputs=base_model.input, outputs=output)


    model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy', F1_score, 'AUC'])


    return model

##########-----------TRAINING/EVALUATING/PREDICTING MODEL-----------##########


def train_model(model, train_gen, valid_gen,  epochs = 50, patience = 1, initial_epoch=0, shuffle=False, validation_steps=None ) :

    es = EarlyStopping(monitor = 'val_accuracy',
                    mode = 'max',
                    patience = 5,s
                    verbose = 1,
                    restore_best_weights = True)

    history=model.fit(x=train_gen,   epochs=50, verbose=1, callbacks=[es],  validation_data=valid_gen,
                validation_steps=validation_steps,  shuffle=shuffle,  initial_epoch=initial_epoch)

    return model, history


def evaluate_model(model, X: np.ndarray, y: np.ndarray, batch_size = 16) :
    '''Evaluate a model.'''

    metrics = model.evaluate(x = X, y = y, batch_size = batch_size, verbose = 1, return_dict = True)

    return metrics

def predict_new(model, X, batch_size = 16) :
    '''Use the selected model to make a prediction.'''

    y_pred = model.predict(x = X, batch_size = batch_size, verbose = 1, callbacks = None)

    return y_pred
