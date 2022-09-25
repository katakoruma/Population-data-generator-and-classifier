
from comet_ml import Experiment
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_decision_forests as tfdf
from datetime import datetime
import sys, os
models = keras.models
layers = keras.layers



class deep_net:
    
    folder = 'models'
    api_key = "NxOMnAPIggVLBb7CuX5qtrvpa"

    time = datetime.now().strftime("%H%M")
    date = datetime.now().strftime("%Y%m%d")

    def __init__(self, task, label=None, dt=True):

        self.dt = dt
        self.label = label
        self.task = task

    def training(self, build_model, imp_train, imp_test, save_path, metrics, process=None):

        model = build_model(task=self.task)
        data_train = pd.read_excel(imp_train, index_col=0)
        data_test = pd.read_excel(imp_test, index_col=0)


        if self.dt:

            if self.task == tfdf.keras.Task.CLASSIFICATION:
                print('set: ', data_train[self.label].values)
                labels = list(set(data_train[self.label])); labels.sort()
                print('labels: ', labels)

                data_train[self.label] = [labels.index(i) for i in data_train[self.label].values]
                data_test[self.label] = [labels.index(i) for i in data_test[self.label].values]

                print(data_train)

            tf_train = tfdf.keras.pd_dataframe_to_tf_dataset(data_train, max_num_classes=len(data_train), label=self.label, task=self.task)
            tf_test = tfdf.keras.pd_dataframe_to_tf_dataset(data_test, max_num_classes=len(data_test), label=self.label, task=self.task)

            #print(np.min(X_train['correction']), np.max(X_train['correction']))

            print('\nCOMPILE\n')
            model.compile(
            metrics=metrics)

            print('\nFIT\n')
            model.fit(tf_train)
            print('\nEVALUATE\n')
            model.evaluate(tf_test)

            print('\nSUMMARY\n')
            print(model.summary())
            print('\nPLOTTER\n')
            tfdf.model_plotter.plot_model_in_colab(model)

            print('\nINSPECTOR\n')
            inspector = model.make_inspector()
            print("Model type:", inspector.model_type())
            print("Number of trees:", inspector.num_trees())
            print("Objective:", inspector.objective())
            print("Input features:", inspector.features())
            print("Tree:",inspector.extract_tree(0))

            model.save(f'{save_path}_{self.label}_{self.date}-{self.time}')

        else:

            model.save(f'{save_path}_{self.date}-{self.time}.h5')

    def prediction(self, model_imp, imp, exp):

        data = pd.read_excel(imp, index_col=0)

        if self.dt:
            model = models.load_model(model_imp)
            X = data.copy()
            del X[self.label]
            print(X)
            
            X = tfdf.keras.pd_dataframe_to_tf_dataset(X,max_num_classes=len(X))

            Y = model.predict(X)
            print(len(Y), Y)
            print(np.min(Y), np.max(Y))

            if self.task == tfdf.keras.Task.CLASSIFICATION:
                labels = list(set(data[self.label])); labels.sort()
                print(labels)

                if Y.shape[1] == 1:
                    Y = np.array([1-Y[:,0], Y[:,0]]).T

                index = tf.argmax(Y, axis=1)
                print(index)
                Y = [labels[i] for i in index]

            data[f'{self.label}_predict'] = Y

        else:
            model = models.load_model(self.path + '%s/%s/%s.h5'%(self.name,self.folder,model_imp))

            X = data.copy()

            Y = model.predict(X).reshape(-1)

        data.to_excel(exp)
        print('***SAVED***')
        print(model_imp, exp)

    def evaluate(self, model_imp, imp):

        model = models.load_model(model_imp)
        data_test = pd.read_excel(imp, index_col=0)

        if self.dt:
            if self.task == tfdf.keras.Task.CLASSIFICATION:
                print('set: ', data_test[self.label].values)
                labels = list(set(data_test[self.label])); labels.sort()
                print('labels: ', labels)

                data_test[self.label] = [labels.index(i) for i in data_test[self.label].values]

                print(data_test)

            tf_test = tfdf.keras.pd_dataframe_to_tf_dataset(data_test, max_num_classes=len(data_test), label=self.label, task=self.task)

        model.evaluate(tf_test)
