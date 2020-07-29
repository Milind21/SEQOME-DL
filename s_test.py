import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import load_model
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score
import seaborn as sns
import os
from pathlib import Path
#path_X=input("Please enter the file location to your feature file:")
#path_Y=input("Please enter the file loaction to your meta/ouptut file:")
if(len(sys.argv)==3):
    path_X=Path(sys.argv[1])
    path_Y=Path(sys.argv[2])
    df_X=pd.read_csv(path_X)
    df_Y=pd.read_csv(path_Y)
    model=tf.keras.models.load_model(str(path_X.parent)+'\\nn.h5')
    col_drop=df_X.columns[0]
    #print(col_drop)
    df_Xf=df_X.drop(col_drop,axis=1)
    df_Yf=df_Y.drop(col_drop,axis=1)

    X_train, X_val, Y_train, Y_val = train_test_split(df_Xf, df_Yf, test_size=0.2, random_state=2)


    X_val=pd.DataFrame(scale(X_val))
    X_train=pd.DataFrame(scale(X_train))
    print("Testing on random subset of dataset:")

    scores = model.evaluate(X_val, Y_val)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
else:
    print("Please enter the feature file and meta file in order!!!")
