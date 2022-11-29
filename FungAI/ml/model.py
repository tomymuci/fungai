from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping


def initialize_model():
    '''whatever'''

    model = models.Sequential()

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(8,(4,4), activation='relu',padding='same',input_shape=(100,100,3)))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(16, (3,3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    model.add(layers.Flatten())


    ### Last layer - Classification Layer with 9 outputs corresponding to 9 mushrooms labels
    model.add(layers.Dense(9,activation='softmax'))

    ### Model compilation
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def train_model(X_train, y_train) :
    '''whatever'''

    model = initialize_model()
    es = EarlyStopping()
    model, history = model.fit(X_train,y_train, batch_size=16, epochs=5, callbacks=[es], verbose=1, validation_split=0.2)

    return model, history
