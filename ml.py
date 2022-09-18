
from comet_ml import Experiment
import pandas as pd
import numpy as np
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

    def __init__(self,):

        pass

    def training(self, build_model, imp_train, imp_test, label, task, save_path, process=None, dt=True):

        model = build_model()
        data_train = pd.read_excel(imp_train, index_col=0)
        data_test = pd.read_excel(imp_test, index_col=0)


        if dt:

            tf_train = tfdf.keras.pd_dataframe_to_tf_dataset(data_train, max_num_classes=len(data_train), label=label, task=task)
            tf_test = tfdf.keras.pd_dataframe_to_tf_dataset(data_test, max_num_classes=len(data_train), label=label, task=task)

            #print(np.min(X_train['correction']), np.max(X_train['correction']))

            print('\nCOMPILE\n')
            model.compile(
            metrics=["mse", "mape"])

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

            model.save(f'{save_path}_{self.date}-{self.time}')

        else:
            pass

            model.save(f'{save_path}_{self.date}-{self.time}.h5')


    def prediction(self,model_imp, exp, dt=False):

        mcluster_list = pd.read_pickle(self.path + '%s/data/%s.p'%(self.name,self.name))

        if dt:
            model = models.load_model(self.path + '%s/%s/%s'%(self.name,self.folder,model_imp))
            X = mcluster_list.loc[:,self.input_list]
            #del X['Cl3D_pt']
            #X.rename(columns = {'Cl3D_pt':'cl3d'}, inplace = True)
            print(X)
            
            X = tfdf.keras.pd_dataframe_to_tf_dataset(X,max_num_classes=len(X))

            Y = model.predict(X).reshape(-1)

            print(np.min(Y), np.max(Y))

        else:
            model = models.load_model(self.path + '%s/%s/%s.h5'%(self.name,self.folder,model_imp))

            X = mcluster_list.loc[:,self.input_list].values.astype('float64')
            X[:,2] = np.abs(X[:,2])

            Y = model.predict(X).reshape(-1)

        if self.res_training:
            mcluster_list['Cl3D_pt'] = Y * mcluster_list['Cl3D_pt']
            mcluster_list['correction'] = Y
        else:
            mcluster_list['Cl3D_pt'] = Y
            Y[Y==0.] = 10**(-9)
            mcluster_list['correction'] = mcluster_list['simpart_pt'].values / Y

        mcluster_list.to_pickle(self.path + '/%s/data/%s.p'%(self.name, exp))
        print('***SAVED***')
        print(model_imp, exp)
