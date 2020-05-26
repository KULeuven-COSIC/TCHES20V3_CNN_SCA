# Revisiting a Methodology for Efficient CNN Architectures in Profiling Attacks

This repository contains examples on how to reproduce the results presented in "Revisiting a Methodology for Efficient CNN Architectures in Profiling Attacks".
The paper is available for download at https://tches.iacr.org/

In this paper we investigate the work "Methodology for Efficient CNN Architectures in Profiling Attacks" by Zaid et al. This was possible, in no small part, because the authors made their code publicly available.
In similar fashion we provide our code to allow others to reproduce our results.



### Colab notebook
For those who would like to explore some of our experiments in an interactive fashion we also provide a Google Colab notebook.
This notebook does not require any special hardware or software setup.
https://colab.research.google.com/drive/1S4ixlEoLm9HqtP3Ku0vqZxm0-S9mq-Cw

The notebook provides an overview and code examples for the experiments carried out in the paper. It includes experiments on varying the filter size and the number of convolutional blocks as well as an example of using the [DeepExplain](https://github.com/marcoancona/DeepExplain) framework to create an attribution map. 


## Repository structure

* ./src/models.py: contains simple functions that define the used CNN models
    * The models proposed by Zaid et al. start with `zaid_`
    * The simplified models in which we removed the first convolutional layer start with `noConv1_` 
* ./src/dataLoaders.py: contains a function to load each of the datasets
    * We opted to pre-compute all possible labels for the attack traces to save time during the network evaluation
    * Each of these functions return:
        * The profiling traces and profiling labels (these will be split into training and validation)
        * The attack traces and labels for each possible key guess
        * The actual value of the target key byte
* ./src/simple.py and multiple.py: see [Examples](##Examples)
* ./models/pretrained_models.tar.xz: see [Pretrained models](##Pretrained-models)         
* ./datasets: This folder contains all of the datasets as provided by Zaid et al.


We provide more details about each dataset in Section 2.1 of our paper.
The original sources for these datasets are:
* DPA-contest v4: http://www.dpacontest.org/v4/42_traces.php
* AES_HD dataset: https://github.com/AESHD/AES_HD_Dataset
* AES_RD dataset: https://github.com/ikizhvatov/randomdelays-traces
* ASCAD: https://github.com/ANSSI-FR/ASCAD

## Examples

### simple.py
This simple example demonstrates how to load a dataset, how to pre-process the data, how to train a model and how to evaluate the trained model.
You can modify this script to load a different dataset and/or train a different model by replacing/modifying the `ascad_100_param` dictionary.

If you have everything setup correctly you should be able to run this script and obtain the following output:
```
$ python3 simple.py 
Using TensorFlow backend.
Creating all target labels for the attack traces:
100%|█████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:04<00:00, 2444.52it/s]
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 700, 1)            0         
_________________________________________________________________
block1_pool (AveragePooling1 (None, 350, 1)            0         
_________________________________________________________________
block2_conv1 (Conv1D)        (None, 350, 64)           3264      
_________________________________________________________________
batch_normalization_1 (Batch (None, 350, 64)           256       
_________________________________________________________________
block2_pool (AveragePooling1 (None, 7, 64)             0         
_________________________________________________________________
block3_conv1 (Conv1D)        (None, 7, 128)            24704     
_________________________________________________________________
batch_normalization_2 (Batch (None, 7, 128)            512       
_________________________________________________________________
block3_pool (AveragePooling1 (None, 3, 128)            0         
_________________________________________________________________
flatten (Flatten)            (None, 384)               0         
_________________________________________________________________
fc1 (Dense)                  (None, 20)                7700      
_________________________________________________________________
fc2 (Dense)                  (None, 20)                420       
_________________________________________________________________
fc3 (Dense)                  (None, 20)                420       
_________________________________________________________________
predictions (Dense)          (None, 256)               5376      
=================================================================
Total params: 42,652
Trainable params: 42,268
Non-trainable params: 384
_________________________________________________________________
Training model: ./models/noConv1_ascad_desync_100.hdf5
During training we will make use of the One Cycle learning rate policy.
Train on 45000 samples, validate on 5000 samples

Epoch 1/50
45000/45000 [==============================] - 5s 121us/step - loss: 5.5821 - acc: 0.0035 - val_loss: 5.5629 - val_acc: 0.0034
Epoch 2/50
45000/45000 [==============================] - 4s 97us/step - loss: 5.5517 - acc: 0.0039 - val_loss: 5.5559 - val_acc: 0.0050

...

Epoch 49/50
45000/45000 [==============================] - 4s 98us/step - loss: 5.2748 - acc: 0.0160 - val_loss: 5.4121 - val_acc: 0.0072
Epoch 50/50
45000/45000 [==============================] - 4s 98us/step - loss: 5.2727 - acc: 0.0154 - val_loss: 5.4119 - val_acc: 0.0074
10000/10000 [==============================] - 0s 46us/step
Evaluating the model:
100%|█████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:05<00:00, 18.94it/s]
```

### multiple.py
An example to demonstrate how you can easily train a model on a dataset using multiple preprocessing strategies.

## Pretrained models
The `pretrained_models.tar.xz` file contains 480 pretrained models. On each of the 6 datasets we trained 2 models (`noConv1_` and `zaid_`) using 4 preprocessing strategies, each experiment was repeated 10 times, the resulting models are provided. More information on this specific experiment can be found in Section 3 of the paper.


## Default parameters
The following parameters should provide decent results, most of them are the same as those provided by Zaid et al.

```
ascad_0_param = {
    'model' : noConv1_ascad_desync_0,
    'nb_epochs' : 50,
    'batch_size' : 50,
    'learning_rate' : 5e-3,
    'one_cycle_lr' : True
}

ascad_50_param = {
    'model' : noConv1_ascad_desync_50,
    'nb_epochs' : 50,
    'batch_size' : 50,
    'learning_rate' : 5e-3,
    'one_cycle_lr' : True
}

ascad_100_param = {
    'model' : noConv1_ascad_desync_100,
    'nb_epochs' : 50,
    'batch_size' : 50,
    'learning_rate' : 5e-3,
    'one_cycle_lr' : True
}

aes_hd_param = {
    'model' : noConv1_aes_hd,
    'nb_epochs' : 20,
    'batch_size' : 256,
    'learning_rate' : 1e-3,
    'one_cycle_lr' : False
}

aes_rd_param = {
    'model' : noConv1_aes_rd,
    'nb_epochs' : 50,
    'batch_size' : 50,
    'learning_rate' : 10e-3,
    'one_cycle_lr' : True
}

dpav4_param = {
    'model' : noConv1_dpav4,
    'nb_epochs' : 50,
    'batch_size' : 50,
    'learning_rate' : 1e-3,
    'one_cycle_lr' : False,
}
```
