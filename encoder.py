import os
import numpy as np
from keras.layers import Input, Reshape, Dense
from keras.layers import Convolution2D, UpSampling2D
from keras.models import Model

class Encoder:
    encoder = None
    autoencoder = None
    model_name = None

    def save(self):
        weights_file_name = '.' + self.model_name + '.h5'
        self.autoencoder.save_weights(weights_file_name)
    
    def load(self):
        weights_file_name = '.' + self.model_name + '.h5'
        if os.path.isfile(weights_file_name):
            print('Found weights file: "' + weights_file_name + '"')
            self.autoencoder.load_weights(weights_file_name) 

    def autoencode(self, tiles):
        X = self.preprocess(tiles)
        Y = self.autoencoder.predict(X)
        return self.postprocess(Y)

    def encode(self, tiles):
        X = self.preprocess(tiles)
        Y = self.encoder.predict(X)
        return Y

    def fit(self, tiles, epochs=1):
        X = self.preprocess(tiles)
        self.autoencoder.fit(X, X, epochs=epochs)

class TopEncoder(Encoder):
    encoder = None
    autoencoder = None
    model_name = 'top_encoder'

    def __init__(self):
        inp = Input(shape=(3, 64, 64))
        hidden = Convolution2D(1, (8, 8), strides=(4, 4), padding='same', activation='relu')(inp)

        encoder = Model(inp, hidden)

        hidden = UpSampling2D((4,4))(hidden)
        hidden = Convolution2D(3, (8, 8), padding='same', activation='sigmoid')(hidden)

        autoencoder = Model(inp, hidden)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        
        self.encoder = encoder
        self.autoencoder = autoencoder

        self.load()

    def preprocess(self, X): return np.array(X) / 255.0
    def postprocess(self, Y): return Y * 255.0

class MidEncoder(Encoder):
    encoder = None
    autoencoder = None
    model_name = 'mid_encoder'

    def __init__(self):
        inp = Input(shape=(1, 32, 32))
        hidden = Convolution2D(1, (8, 8), strides=(4, 4), padding='same', activation='relu')(inp)

        encoder = Model(inp, hidden)

        hidden = UpSampling2D((4,4))(hidden)
        hidden = Convolution2D(1, (8, 8), padding='same', activation='sigmoid')(hidden)

        autoencoder = Model(inp, hidden)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        
        self.encoder = encoder
        self.autoencoder = autoencoder

        self.load()

    def preprocess(self, X): return np.array(X)
    def postprocess(self, Y): return Y

class BumEncoder(Encoder):
    encoder = None
    autoencoder = None
    model_name = 'bum_encoder'

    def __init__(self):
        inp = Input(shape=(1, 8, 8))
        hidden = Reshape((64,))(inp)
        hidden = Dense(16, activation='relu')(hidden)
        hidden = Dense(4, activation='relu')(hidden)

        encoder = Model(inp, hidden)

        hidden = Dense(16, activation='relu')(hidden)
        hidden = Dense(64, activation='sigmoid')(hidden)
        hidden = Reshape((1, 8, 8))(hidden)

        autoencoder = Model(inp, hidden)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        
        self.encoder = encoder
        self.autoencoder = autoencoder

        self.load()

    def preprocess(self, X): return np.array(X)
    def postprocess(self, Y): return Y

