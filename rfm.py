
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
metrics = ["mse","binary_crossentropy", "categorical_crossentropy", "accuracy"]
metrics = ["mse", "sparse_categorical_accuracy", "accuracy"]
#########
#PREDICTION

model_imp = 'models/rfm_education_20220925-0340'
imp =  'data/population_data.xlsx'
exp = 'data/population_data_predict.xlsx'

########
label = 'education'

task = tfdf.keras.Task.CLASSIFICATION
#task = tfdf.keras.Task.REGRESSION


def build_model(task, hp=None):
    model = tfdf.keras.RandomForestModel(task=task, num_trees=8, max_depth=16,)
    return model


if __name__ == '__main__':

    mode = 't'

    dn = deep_net(task, label, dt=True)

    if len(sys.argv) != 1:
        argv = sys.argv[1]
    else:
        argv = None
    
    if argv == 't' or mode == 't':
        print('Training')
        if len(sys.argv) > 2:
            dn.training(build_model, imp_train, imp_test, save_path, metrics, process=sys.argv[2])
        else:
            dn.training(build_model, imp_train, imp_test, save_path, metrics)

    elif argv == 'p' or mode == 'p':
        print('Prediction')
        dn.prediction(model_imp, imp, exp)
    elif argv == 'e' or mode == 'e':
        print('Evaluate')
        dn.evaluate(model_imp, imp)
    else:
        print('Miep')
    