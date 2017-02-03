"""
Builds a manifold learning autoencoders.

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
from keras.models import Sequential
from keras.optimizers import Adagrad, SGD, Adadelta, Adam
from keras.regularizers import l2
from keras.layers import Input, Dense
from keras.models import Model
#from early_stopping import MyEarlyStopping
from stacked_ae import save_decoder
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from util import pick_k

        
def train_decoder(inputs, outputs, model, lr, epsilon, weights=None, nb_epoch=1000, loss='mse', verbose=False):
    
    if weights is not None:
        model.set_weights(weights)
    
    # training
#    optimizer = SGD(lr=lr, momentum=momentum, decay=lr_decay, nesterov=True)
    optimizer = Adagrad(lr=lr, epsilon=epsilon)
#    optimizer = Adadelta(lr=lr, rho=rho, epsilon=epsilon)
    model.compile(loss=loss, optimizer=optimizer)
#    early_stopping = MyEarlyStopping(monitor='loss', patience=10, verbose=verbose, tol=1e-6)
    model.fit(inputs, outputs, batch_size=inputs.shape[0], nb_epoch=nb_epoch, verbose=verbose)#, callbacks=[early_stopping])

    return model
    
def mlae(data, feature_dim, train, test, hidden_size_l1=0, hidden_size_l2=0, hidden_size_l3=0, hidden_size_l4=0, 
         l=0, lr=0.01, epsilon=1e-08, evaluation=False, overwrite=True):
    ''' Select number of layers for autoencoder based on arguments 
        hidden_size_l1, hidden_size_l2, hidden_size_l3 and hidden_size_l4 '''
    
    np.random.seed(0)
    
    # Encoder
    k_opt = pick_k(data[train], feature_dim)
#    encoder = LocallyLinearEmbedding(n_neighbors=k_opt, n_components=feature_dim, method='hessian').fit(data[train])
    encoder = Isomap(n_neighbors=k_opt, n_components=feature_dim).fit(data[train])
    
    features = np.zeros((data.shape[0],feature_dim))
    features[train+test] = encoder.transform(data[train+test])
    
    # Decoder
    verbose = 0
    activation = 'tanh'
    loss = 'mse'
    nb_epoch = 5000 # maximum number of epochs

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
    
    inputs = Input(shape=(feature_dim,))
    x = inputs
    for i in range(n_layers):
        x = Dense(sizes[-i-2], activation=activation, W_regularizer=l2(l))(x)
    decoded = x
    
    model = Model(input=inputs, output=decoded)
    model = train_decoder(features[train], data[train], model, lr, epsilon, nb_epoch=nb_epoch, loss=loss, verbose=verbose)
    
    if evaluation:
        # Used for hyperparameter optimization
        cost = model.evaluate(features[test], data[test], batch_size=len(test), verbose=verbose)
        return cost
    
    # Reconstruct using the decoder
    decoder = Sequential()
    for i in range(n_layers):
        decoder.add(Dense(sizes[-i-2], input_dim=sizes[-i-1], activation=activation, 
                          weights=model.layers[-n_layers+i].get_weights()))
    decoder.compile(loss='mse', optimizer='sgd')

    name = 'MLAE-'+str(n_layers)
    
    if overwrite:
        # Save the decoder
        save_decoder(decoder, len(train), name)
    
    return features, name, decoder.predict

