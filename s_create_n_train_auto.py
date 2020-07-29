import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow import keras
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.preprocessing import scale
from pathlib import Path

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#path_X=input("Please enter the file location to your feature file:")
#path_Y=input("Please enter the file loaction to your meta/ouptut file:")
if(len(sys.argv)==3):

    #read the csv files
    path_X=Path(sys.argv[1])
    path_Y=Path(sys.argv[2])
    df_X=pd.read_csv(path_X)
    df_Y=pd.read_csv(path_Y)

    df_X=df_X.drop(df_X.columns[0],axis=1)
    df_Y=df_Y.drop(df_Y.columns[0],axis=1)

    ip_dim=ip_node=len(df_X.columns)
    op_node=int(df_Y.nunique(axis=0))
    ##create a dense layer for input
    model=Sequential()
    model.add(Dense(ip_node, input_dim=ip_dim, activation='relu'))

    layer_count=1
    X_train, X_val, Y_train, Y_val = train_test_split(df_X, df_Y, test_size=0.2, random_state=2)
    X_train=pd.DataFrame(scale(X_train))
    X_val=pd.DataFrame(scale(X_val))
    log_dir=f"log{int(time.time())}"
    def build_model(hyp_param):
        model = Sequential()
        model.add(Dense(hyp_param.Int("input_units",2,80,4), input_dim=X_train.shape[1], activation='relu'))
        for i in range(hyp_param.Int("n_layers",1,5)):
            model.add(Dense(hyp_param.Int(f"{i}_units",2,80,4), activation='relu'))
        model.add(Dense(op_node,activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        return model
    tuner = RandomSearch(
        build_model,
        objective='categorical_accuracy',
        max_trials= 5,
        executions_per_trial=1,
        directory=log_dir
        )
    tuner.search(x=X_train,
                 y=Y_train,
                 epochs=5,
                 batch_size=10,
                 validation_data=(X_val,Y_val)
                 )
    model = tuner.get_best_models(num_models=1)[0]

    opt=tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    # Fit the model
    model.fit(X_train, Y_train, epochs=64, batch_size=3,validation_data=(X_val, Y_val))
    print("\nFollowing is the summary of the model generated:")
    model.summary()
    save_path=str(path_X.parent)
    model.save(save_path+'\\nn.h5')
    plot_model(model, to_file='model_plot_img.png', show_shapes=True, show_layer_names=True)
    img=mpimg.imread('model_plot_img.png')
    imgplot = plt.imshow(img)
    print("\nFollowing is the model diagram:")

    plt.show()
    #save model

else:
    print("Please enter the feature file and meta file in order!!!")
