
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

imp_train = 'export/training_data.xlsx'
imp_test = 'export/test_data.xlsx'

save_path = 'models/bdt'

label = 'age'

task = tfdf.keras.Task.CLASSIFICATION
task = tfdf.keras.Task.REGRESSION


def build_model(hp=None):
    model = tfdf.keras.GradientBoostedTreesModel(task=tfdf.keras.Task.REGRESSION, num_trees=512, max_depth=32,)
    return model


if __name__ == '__main__':

    dn = deep_net()

    if sys.argv[1] == 't':
        print('Training')
        if len(sys.argv) > 2:
            dn.training(build_model, imp_train, imp_test, label, task, save_path, process=sys.argv[2], dt=True)
        else:
            dn.training(build_model, imp_train, imp_test, label, task, save_path, dt=True)

    elif sys.argv[1] == 'p':
        print('Prediction')
        dn.prediction()
    else:
        print('Miep')