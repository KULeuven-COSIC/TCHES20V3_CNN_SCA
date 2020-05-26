import h5py
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from clr import OneCycleLR
import dataLoaders
import preproces
import models
from tqdm import tqdm

def unpackParams(p):
    return p['model'], p['nb_epochs'], p['learning_rate'], p['batch_size'], p['one_cycle_lr']


def trainNetwork(params, tracesTrain, tracesVal, labelsTrain, labelsVal):
    mod, nb_epochs, learning_rate, batch_size, one_cycle_lr = unpackParams(params)
    
    K.clear_session()
    model = mod(input_size=tracesTrain.shape[1], learning_rate=learning_rate)
    model.summary()
        
    # Ensure the data is in the right shape
    input_layer_shape = model.get_layer(index=0).input_shape
    if len(input_layer_shape) == 2:
        tracesTrain_shaped = tracesTrain
        tracesVal_shaped = tracesVal
    elif len(input_layer_shape) == 3:
        tracesTrain_shaped = tracesTrain.reshape((tracesTrain.shape[0], tracesTrain.shape[1], 1))
        tracesVal_shaped = tracesVal.reshape((tracesVal.shape[0], tracesVal.shape[1], 1))

    modelpath = './../models/' + mod.__name__ + '.hdf5'
    print('Training model:', modelpath)
    checkpoint = ModelCheckpoint(modelpath, verbose=0, save_best_only=False)

    if one_cycle_lr:
        print('During training we will make use of the One Cycle learning rate policy.')
        lr_manager = OneCycleLR(max_lr=learning_rate, end_percentage=0.2, scale_percentage=0.1, maximum_momentum=None, minimum_momentum=None, verbose=False)
        callbacks = [checkpoint, lr_manager]
    else:
        callbacks = [checkpoint]

    history = model.fit(x=tracesTrain_shaped, y=to_categorical(labelsTrain, num_classes=256), validation_data=(tracesVal_shaped, to_categorical(labelsVal, num_classes=256)), batch_size=batch_size, verbose=1, epochs=nb_epochs, callbacks=callbacks)
    return history


def rank(predictions, key, targets, ntraces, interval=10):
    ranktime = np.zeros(int(ntraces/interval))
    pred = np.zeros(256)

    idx = np.random.randint(predictions.shape[0], size=ntraces)
    
    for i, p in enumerate(idx):
        for k in range(predictions.shape[1]):
            pred[k] += predictions[p, targets[p, k]]
            
        if i % interval == 0:
            ranked = np.argsort(pred)[::-1]
            ranktime[int(i/interval)] = list(ranked).index(key)
            
    return ranktime


# Load a dataset (see dataLoaders.py)
X_profiling, Y_profiling, X_attack, targets, key_attack = dataLoaders.load_ascad('./../datasets/ASCAD_dataset/ASCAD_desync100.h5')

# Preprocess the data (see preproces.py)
X_profiling_processed, X_attack_processed = preproces.horizontal_standardization(X_profiling, X_attack)

# Automatic split (90/10) 
# Random_state is set to get reproducible splits
# model weights are initialised randomly (can also saved in case we encouter weird things)
tracesTrain, tracesVal, labelsTrain, labelsVal = train_test_split(X_profiling_processed, Y_profiling, test_size=0.1, random_state=0)
 
# Define training parameters (see models.py for the predefined models)
ascad_100_param = {
    'model' : models.noConv1_ascad_desync_100,
    'nb_epochs' : 50,
    'batch_size' : 50,
    'learning_rate' : 5e-3,
    'one_cycle_lr' : True
}

hist = trainNetwork(ascad_100_param, tracesTrain, tracesVal, labelsTrain, labelsVal)

# Model evaluation
ntraces = 300 # Number of traces to use per attack attempt
nattack = 100 # Number of attacks, each using ntraces traces
interval = 1  # granualirity

modelpath = './../models/' + ascad_100_param['model'].__name__ + '.hdf5'
model = load_model(modelpath)

input_layer_shape = model.get_layer(index=0).input_shape
if len(input_layer_shape) == 2:
    tracesAttack_shaped = X_attack_processed
elif len(input_layer_shape) == 3:
    tracesAttack_shaped = X_attack_processed.reshape((X_attack_processed.shape[0], X_attack_processed.shape[1], 1))

predictions = model.predict(tracesAttack_shaped, verbose=1)
predictions = np.log(predictions+1e-40)

print('Evaluating the model:')
ranks = np.zeros((nattack, int(ntraces/interval)))
for i in tqdm(range(nattack)):
    ranks[i] = rank(predictions, key_attack, targets, ntraces, interval)

# Calculate the mean of the rank over the nattack attacks
result = np.mean(ranks, axis=0)

# Simple plot of the result
plt.plot(result)
plt.savefig('./../models/' + ascad_100_param['model'].__name__ + '.png')
