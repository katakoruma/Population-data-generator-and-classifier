
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


def class_to_num(df, key=None, replace_all=False):
    
    if replace_all:

        for key in df.keys():
            if type(df.at[0,key]) == str:
                df = class_to_num(df, key=key, replace_all=False)

    else:
        #print('set: ', df[key].values)
        labels = list(set(df[key])); labels.sort()
        print(f'labels for {key}: ', labels)

        df[key] = [labels.index(i) for i in df[key].values]
        print(df)


    return df

class deep_net:
    
    folder = 'models'
    api_key = "NxOMnAPIggVLBb7CuX5qtrvpa"

    time = datetime.now().strftime("%H%M")
    date = datetime.now().strftime("%Y%m%d")

    def __init__(self, task, label=None, dt=True):

        self.dt = dt
        self.label = label
        self.task = task

    def training(self, build_model, imp_train, imp_test, save_path, metrics, lr=None, process=None):

        model = build_model(task=self.task)
        data_train = pd.read_excel(imp_train, index_col=0)
        data_test = pd.read_excel(imp_test, index_col=0)

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

        print('\nSUMMARY\n')
        print(model.summary())
        print('\nPLOTTER\n')
        tfdf.model_plotter.plot_model_in_colab(model)

        print('\nEVALUATE\n')
        model.evaluate(tf_test)

        print('\nINSPECTOR\n')
        inspector = model.make_inspector()
        print("Model type:", inspector.model_type())
        print("Number of trees:", inspector.num_trees())
        print("Objective:", inspector.objective())
        print("Input features:", inspector.features())
        print("Tree:",inspector.extract_tree(0))

        model.save(f'{save_path}_{self.label}_{self.date}-{self.time}')


    def train_deep(self, build_model, imp_train, imp_test, save_path, metrics, batch_size, epochs, lr, early_stop_patience, validation_split, process=None):

        data_train = pd.read_excel(imp_train, index_col=0)
        data_test = pd.read_excel(imp_test, index_col=0)

        data_train = class_to_num(data_train, replace_all=True)
        data_test =  class_to_num(data_test, replace_all=True)

        X_train  = data_train.copy()
        Y_train = X_train.pop(self.label)
        Y_train = tf.one_hot(Y_train,len(list(set(Y_train)))).numpy()

        X_test  = data_test.copy()
        Y_test = X_test.pop(self.label)
        Y_test = tf.one_hot(Y_test,len(list(set(Y_test)))).numpy()

        X_train, X_test = X_train.values, X_test.values

        print('X_train : ',X_train.shape)
        print('Y_train : ',Y_train.shape)

        model = build_model(X_train, Y_train)

        model.compile(
            loss="mean_squared_error",
            optimizer=keras.optimizers.Adam(lr=lr),
            metrics=metrics)

        print(model.summary())

        #X_train, Y_train, X_test, Y_test = X_train.T, Y_train.T, X_test.T, Y_test.T

        print('X_train : ',X_train)
        print('Y_train : ',Y_train)

        model.fit(
            X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_split=validation_split, 
            callbacks=[keras.callbacks.EarlyStopping(patience=early_stop_patience)])

        model.evaluate(X_test,Y_test)


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
                Y_p = [labels[i] for i in index]

                data[f'{self.label}_predict'] = Y_p
                data[f'{self.label}_predict_p'] = [Y[i,:] for i in range(Y.shape[0])]
            
            else:

                data[f'{self.label}_predict'] = Y

        else:

            model = models.load_model(model_imp)

            X = data.copy()
            Y = model.predict(X).reshape(-1)

        data.to_excel(exp)
        print('***SAVED***')
        print(model_imp, exp)

    def predict_deep():
        pass

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
