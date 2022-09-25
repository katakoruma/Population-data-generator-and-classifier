from comet_ml import Experiment
import pandas as pd
import numpy as np
from tensorflow import keras
import sys, os
import tensorflow_decision_forests as tfdf
import keras_tuner as kt
from ml import deep_net
models = keras.models
layers = keras.layers
backend = keras.backend

#########
#TRAINING

imp_train = 'data/training_data.xlsx'
imp_test = 'data/test_data.xlsx'

save_path = 'models/rfm'

metrics = ["mse", "mape"]


batch_size = 1024
epochs = 2000
validation_split = 0.2
early_stop_patience = 30
lr = 1e-3
#########
#PREDICTION

model_imp = 'models/rfm_education_20220925-0340'
imp =  'data/population_data.xlsx'
exp = 'data/population_data_predict.xlsx'

########
label = 'education'

task = tfdf.keras.Task.CLASSIFICATION
#task = tfdf.keras.Task.REGRESSION

def build_model(input_list):
    normal = layers.BatchNormalization(input_shape=(len(input_list),))
    layer1 = layers.Dense(4, activation='relu') 
    layer2 = layers.Dense(1, activation='relu')

    model = models.Sequential([normal,layer1,layer2])

    return model


if __name__ == '__main__':

    mode = 't'

    dn = deep_net(task, label, dt=False)

    if len(sys.argv) != 1:
        argv = sys.argv[1]
    else:
        argv = None
    
    if argv == 't' or mode == 't':
        print('Training')
        if len(sys.argv) > 2:
            dn.train_deep(build_model, imp_train, imp_test, save_path, metrics, batch_size, epochs, lr, validation_split, process=sys.argv[2])
        else:
            dn.train_deep(build_model, imp_train, imp_test, save_path, metrics, batch_size, epochs, lr, validation_split)

    elif argv == 'p' or mode == 'p':
        print('Prediction')
        dn.prediction(model_imp, imp, exp)
    elif argv == 'e' or mode == 'e':
        print('Evaluate')
        dn.evaluate(model_imp, imp)
    else:
        print('Miep')