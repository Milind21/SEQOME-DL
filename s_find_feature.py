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
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score
import seaborn as sns
from pathlib import Path
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

    X_train, X_val, Y_train, Y_val = train_test_split(df_X, df_Y, random_state=2)
    X_train=pd.DataFrame(scale(X_train))
    X_val=pd.DataFrame(scale(X_val))
    model_reg = LogisticRegression(max_iter=1000).fit(X_train, Y_train.values.ravel())
    reg_pred=model_reg.predict(X_val)
    acc_base=accuracy_score(Y_val.values.ravel(), reg_pred)
    roc_auc_base=roc_auc_score(Y_val.values.ravel(), reg_pred)
    print("\nAccuracy with all features is={}".format(acc_base))
    print("\nROC AUC Score with all features is={}".format(roc_auc_base))
    acc={}
    roc_auc={}
    cols=df_X.columns
    for i in range(len(df_X.columns)):
        col=cols[i]
        #print(col)
        temp_df=df_X
        change_col=temp_df[col]
        #print(change_col)
        shuffled_col=np.random.permutation(change_col)
        #print(temp_df.columns)
        temp_df=temp_df.drop(col,axis=1)
        #print(temp_df.columns)
        #temp_df[col]=list(np.random.permutation(change_col))
        temp_df.insert(loc=i, column=col, value=shuffled_col)
        #print(temp_df.columns)
        X_train_temp,X_val_temp,Y_train_temp,Y_val_temp=train_test_split(temp_df,df_Y,random_state=2)
        reg_pred_temp=model_reg.predict(X_val_temp)
        accu=accuracy_score(Y_val.values.ravel(), reg_pred_temp)
        roc_aucc=roc_auc_score(Y_val.values.ravel(), reg_pred_temp)

        acc.update({col:accu})
        roc_auc.update({col:roc_aucc})
    print("\nAccuracy with following variable removed:")
    print(acc)
    print("\nROC AUC Score with following variable removed:")
    print(roc_auc)
    print("The p-value obtained from F statistics (ANOVA test) is often used for feature selection")
    sel=f_classif(X_train,Y_train)
    p_values=pd.Series(sel[1])
    p_values.index = X_train.columns
    p_values.sort_values(ascending=True,inplace=True)
    print(p_values)
    threshold = input("Please enter the threshold for significance level (in decimals e.g. 0.05) : ")
    cols=p_values[p_values<float(threshold)].index
    print("There are "+ str(len(cols)) + " number of columns with threshold less than " + str(threshold) + " and the cols are" + str(cols))
    print("\nYou can also see the correlation matrix and pairplot of all the columns if you wish to remove some correlated variables:")
    df=pd.concat([df_X, df_Y], axis=1)
    corr=df.corr()
    plt.figure(figsize=(8,8))
    corr_plot=sns.heatmap(corr,annot=True,cmap="coolwarm")
    fig_corr = corr_plot.get_figure()
    fig_corr.savefig(str(path_X.parent)+"\corr_matrix.png")
    pair_plot=sns.pairplot(df,height=1.5)
    pair_plot.savefig(str(path_X.parent)+"\pair_plot.png")
    plt.show()

else:
    print("Please enter the feature file and meta file in order!!!")
