"""
Builds a stacked autoencoder.

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
from keras.models import Sequential
from keras.optimizers import Adagrad, SGD, Adadelta, Adam
from keras.regularizers import l2
from keras.layers import Input, Dense, noise
from keras.models import Model
from keras import backend as K
#from early_stopping import MyEarlyStopping
import ConfigParser

def save_decoder(model, ae_id, c):
    
    # Get the file name
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    source = config.get('Global', 'source')
    noise_scale = config.getfloat('Global', 'noise_scale')
    
    if source == 'sf':
        alpha = config.getfloat('Superformula', 'nonlinearity')
        beta = config.getint('Superformula', 'n_clusters')
        sname = source + '-' + str(beta) + '-' + str(alpha)
        
    elif source == 'glass' or source[:3] == 'sf-':
        sname = source
        
    fname = '%s_%.4f_%s_%d' % (sname, noise_scale, ae_id, c)
    
    # Save model architecture and weights
    json_string = model.to_json()
    open('./trained_models/'+fname+'_architecture.json', 'w').write(json_string)
    model.save_weights('./decoders/'+fname+'_weights.h5', overwrite=True)
    
def train_ae(data, feature_dim, hidden_sizes, l, p=0, batch_size=100, activation='tanh', 
             activity_regularizer=None, weights=None, nb_epoch=1000, loss='mse', verbose=False):

    data_dim = data.shape[1]
    inputs = Input(shape=(data_dim,))
    
    sizes = [data_dim] + hidden_sizes + [feature_dim]
    n_layers = len(sizes) - 1
    
    # Encoder
    x = noise.GaussianDropout(p)(inputs)
    for i in range(n_layers):
        x = Dense(sizes[i+1], activation=activation, W_regularizer=l2(l))(x)
    
    # Decoder
    for i in range(n_layers):
        x = Dense(sizes[-i-2], activation=activation, W_regularizer=l2(l))(x)
    decoded = x
    
    model = Model(input=inputs, output=decoded)
    
    if weights is not None:
        model.set_weights(weights)
        
#    optimizer = Adagrad(lr=lr, epsilon=epsilon)
    optimizer = Adam()
    model.compile(loss=loss, optimizer=optimizer)
#    early_stopping = MyEarlyStopping(monitor='loss', patience=10, verbose=verbose, tol=1e-6)
    model.fit(data, data, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)#, callbacks=[early_stopping])
    
    if n_layers == 1:
        W_en = model.layers[-2].get_weights()
        W_de = model.layers[-1].get_weights()
    else:
        W_en = None
        W_de = None
        
    encode = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-2].output])
    a = encode([data, 0])[0] # hidden layer's activation
    
    return a, W_en, W_de, model
    
def sae(data, c, feature_dim, train, test, hidden_size_l1=0, hidden_size_l2=0, hidden_size_l3=0, hidden_size_l4=0, p=0.3,
        l=0, batch_size=100, evaluation=False, overwrite=True):
    ''' Select number of layers for autoencoder based on arguments 
        hidden_size_l1, hidden_size_l2, hidden_size_l3 and hidden_size_l4 '''
    
    np.random.seed(0)
    
    pre_training = False
    verbose = 0
    activation = 'tanh'
    loss = 'mse'
    nb_epoch = 5000 # maximum number of epochs
#    p = 0.1 # dropout fraction for denoising autoencoders

    if hidden_size_l1 == 0:
        hidden_sizes = []
    elif hidden_size_l2 == 0:
        hidden_sizes = [hidden_size_l1]
    elif hidden_size_l3 == 0:
        hidden_sizes = [hidden_size_l1, hidden_size_l2]
    elif hidden_size_l4 == 0:
        hidden_sizes = [hidden_size_l1, hidden_size_l2, hidden_size_l3]
    else:
        hidden_sizes = [hidden_size_l1, hidden_size_l2, hidden_size_l3, hidden_size_l4]
    
    data_dim = data.shape[1]
    sizes = [data_dim] + hidden_sizes + [feature_dim]
    n_layers = len(sizes) - 1
    Ws = None
    
    # Pre-training (greedy layer-wise training)
    if pre_training:
        Ws_en = []
        Ws_de = []
        a = data[train]
        for i in range(n_layers):
            if verbose:
                print 'Pre-training for Layer %d ...' % (i+1)
            a, W_en, W_de, _ = train_ae(a, sizes[i+1], [], l, p=p, batch_size=batch_size,
                                        nb_epoch=nb_epoch, loss=loss, verbose=verbose)
            Ws_en.append(W_en)
            Ws_de.append(W_de)
        
        Ws_de.reverse()
        Ws = Ws_en + Ws_de
        Ws = [item for sublist in Ws for item in sublist]
    
    # Fine tuning
    if verbose:
        print 'Fine tuning ...'
    _, _, _, model = train_ae(data[train], feature_dim, hidden_sizes, l, p=p, batch_size=batch_size,
                              nb_epoch=nb_epoch, loss=loss, verbose=verbose, weights=Ws)
    
    if evaluation:
        # Used for hyperparameter optimization
        cost = model.evaluate(data[test], data[test], batch_size=len(test), verbose=verbose)
        return cost
    
    # Reconstruct using the decoder
    decoder = Sequential()
    for i in range(n_layers):
        decoder.add(Dense(sizes[-i-2], input_dim=sizes[-i-1], activation=activation, 
                          weights=model.layers[-n_layers+i].get_weights()))
    decoder.compile(loss='mse', optimizer='sgd')

    if p > 0:
        name = 'SDAE-'+str(n_layers)
    else:
        name = 'SAE-'+str(n_layers)
    
    if overwrite:
        # Save the decoder
        save_decoder(decoder, name, c)
    
    encode = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-n_layers-1].output])
    features = np.zeros((data.shape[0],feature_dim))
    features[train+test] = encode([data[train+test], 0])[0]
    
    return features, name, decoder.predict

