import os
import numpy as np
from keras.layers import Input, Reshape, Dense
from keras.layers import LSTM
from keras.models import Model



class QScore:
    lstm = None
    maxlen = None
    model_name = 'q_score'

    def __init__(self, maxlen, input_length):
        inp = Input(shape=(maxlen, input_length))
        
        hidden = LSTM(64)(inp)
        hidden = Dense(1)(hidden)

        lstm = Model(inp, hidden)
        lstm.compile(loss='mean_squared_error', optimizer='rmsprop')

        self.lstm = lstm

        self.load()

    def save(self):
        weights_file_name = '.' + self.model_name + '.h5'
        self.lstm.save_weights(weights_file_name)
    
    def load(self):
        weights_file_name = '.' + self.model_name + '.h5'
        if os.path.isfile(weights_file_name):
            print('Found weights file: "' + weights_file_name + '"')
            self.lstm.load_weights(weights_file_name)
    
    def fit(self, X, Y, epochs=1):
        X = np.array(X)
        Y = np.array(Y)
        self.lstm.fit(X, Y, epochs=epochs)

    def predict(self, X):
        X = np.array(X)
        return self.lstm.predict(X)
