#!/usr/bin/env python
# coding: utf-8

# # Data Processor - Financial Data:
# - AAPL
# - Bitcoin

# ### Auxiliary Initializations

# In[11]:


#!/usr/bin/env python
# coding: utf-8

# # Initializations
# Here we dump the list of all initializations needed to run any code snippet for the NEU.

# ---

# In[ ]:

# (Semi-)Classical Regressor(s)
from scipy.interpolate import interp1d
import statsmodels.api as sm
import rpy2.robjects as robjects # Work directly from R (since smoothing splines packages is better)
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV

# (Semi-)Classical Dimension Reducers
from sklearn.decomposition import PCA


# Grid-Search and CV
from sklearn.model_selection import RandomizedSearchCV, KFold
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# Data Structuring
import numpy as np
import pandas as pd


# Pre-Processing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from scipy.special import expit

# Regression
from sklearn.linear_model import LinearRegression
from scipy import linalg as scila

# Random Forest & Gradient Boosting (Arch. Construction)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Random-Seeds
import random

# Rough Signals
from fbm import FBM

# Tensorflow
import tensorflow as tf
from keras.utils.layer_utils import count_params
import keras as K
import keras.backend as Kb
from keras.layers import *
from keras.models import Model
from keras.models import Sequential
from keras import layers
from keras import utils as np_utils
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.optimizers import Adam

# Operating-System Related
import os
from pathlib import Path
import pickle
#from sklearn.externals import joblib

# Timeing
import time
import datetime as DT

# Visualization
import matplotlib
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns; sns.set()

# z_Misc
import math
import warnings













########################
# Make Paths
########################

Path('./outputs/models/').mkdir(parents=True, exist_ok=True)
Path('./outputs/models/Benchmarks/').mkdir(parents=True, exist_ok=True)
Path('./outputs/models/Benchmarks/Invertible_Networks/GLd_Net/').mkdir(parents=True, exist_ok=True)
Path('./outputs/models/Benchmarks/Invertible_Networks/Ed_Net/').mkdir(parents=True, exist_ok=True)
Path('./outputs/models/Benchmarks/Linear_Regression/').mkdir(parents=True, exist_ok=True)
Path('./outputs/models/NAIVE_NEU/').mkdir(parents=True, exist_ok=True)
Path('./outputs/models/NEU/').mkdir(parents=True, exist_ok=True)
Path('./outputs/tables/').mkdir(parents=True, exist_ok=True)
Path('./outputs/results/').mkdir(parents=True, exist_ok=True)
Path('./outputs/plotsANDfigures/').mkdir(parents=True, exist_ok=True)
Path('./inputs/data/').mkdir(parents=True, exist_ok=True)


## Melding Parameters (These are put to meet rapid ICML rebuttle deadline when merging codes)
Train_step_proportion = test_size_ratio


# ### Prepare Data

# In[12]:


#------------------------#
# Run External Notebooks #
#------------------------#
if Option_Function == "SnP" or "AAPL":
    #--------------#
    # Get S&P Data #
    #--------------#
    #=# SnP Constituents #=#
    # Load Data
    snp_data = pd.read_csv('inputs/data/snp500_data/snp500-adjusted-close.csv')
    # Format Data
    ## Index by Time
    snp_data['date'] = pd.to_datetime(snp_data['date'],infer_datetime_format=True)
    #-------------------------------------------------------------------------------#

    #=# SnP Index #=#
    ## Read Regression Target
    snp_index_target_data = pd.read_csv('inputs/data/snp500_data/GSPC.csv')
    ## Get (Reference) Dates
    dates_temp = pd.to_datetime(snp_data['date'],infer_datetime_format=True).tail(600)
    ## Format Target
    snp_index_target_data = pd.DataFrame({'SnP_Index': snp_index_target_data['Close'],'date':dates_temp.reset_index(drop=True)})
    snp_index_target_data['date'] = pd.to_datetime(snp_index_target_data['date'],infer_datetime_format=True)
    snp_index_target_data.set_index('date', drop=True, inplace=True)
    snp_index_target_data.index.names = [None]
    #-------------------------------------------------------------------------------#
    
    ## Get Rid of Rubbish
    snp_data.set_index('date', drop=True, inplace=True)
    snp_data.index.names = [None]
    ## Get Rid of NAs and Expired Trends
    snp_data = (snp_data.tail(600)).dropna(axis=1).fillna(0)
    
    #---------------------------------------------------------------------------------------------------#
    #---------------------------------------------------------------------------------------------------#
    # Apple
    #---------------------------------------------------------------------------------------------------#
    #---------------------------------------------------------------------------------------------------#
    if Option_Function == "AAPL":
        snp_index_target_data = snp_data[{'AAPL'}]
        snp_data = snp_data[{'IBM','QCOM','MSFT','CSCO','ADI','MU','MCHP','NVR','NVDA','GOOGL','GOOG'}]
    #---------------------------------------------------------------------------------------------------#
    #---------------------------------------------------------------------------------------------------#
    
    # Get Return(s)
    snp_data_returns = snp_data.diff().iloc[1:]
    snp_index_target_data_returns = snp_index_target_data.diff().iloc[1:]
    #--------------------------------------------------------#
    
    #-------------#
    # Subset Data #
    #-------------#
    # Get indices
    N_train_step = int(round(snp_index_target_data_returns.shape[0]*Train_step_proportion,0))
    N_test_set = int(snp_index_target_data_returns.shape[0] - round(snp_index_target_data_returns.shape[0]*Train_step_proportion,0))
    # # Get Datasets
    X_train = snp_data_returns[:N_train_step]
    X_test = snp_data_returns[-N_test_set:]
    ## Coerce into format used in benchmark model(s)
    data_x = X_train
    data_x_test = X_test
    # Get Targets 
    data_y = snp_index_target_data_returns[:N_train_step]
    data_y_test = snp_index_target_data_returns[-N_test_set:]
    
    # Scale Data
    scaler = StandardScaler()
    data_x = scaler.fit_transform(data_x)
    data_x_test = scaler.transform(data_x_test)

    # # Update User
    print('#================================================#')
    print(' Training Datasize: '+str(X_train.shape[0])+' and test datasize: ' + str(X_test.shape[0]) + '.  ')
    print('#================================================#')

    # # Set First Run to Off
    First_run = False

    #-----------#
    # Plot Data #
    #-----------#
    fig = snp_data_returns.plot(figsize=(16, 16))
    fig.get_legend().remove()
    plt.title("S&P Data Returns")

    # SAVE Figure to .eps
    plt.savefig('./outputs/plotsANDfigures/SNP_Data_returns.pdf', format='pdf')


# In[ ]:


#------------------------#
# Run External Notebooks #
#------------------------#
if Option_Function == "crypto":
    #--------------#
    # Prepare Data #
    #--------------#
    # Read Dataset
    crypto_data = pd.read_csv('inputs/data/cryptocurrencies/Cryptos_All_in_one.csv')
    # Format Date-Time
    crypto_data['Date'] = pd.to_datetime(crypto_data['Date'],infer_datetime_format=True)
    crypto_data.set_index('Date', drop=True, inplace=True)
    crypto_data.index.names = [None]

    # Remove Missing Data
    crypto_data = crypto_data[crypto_data.isna().any(axis=1)==False]

    # Get Returns
    crypto_returns = crypto_data.diff().iloc[1:]

    # Parse Regressors from Targets
    ## Get Regression Targets
    crypto_target_data = pd.DataFrame({'BITCOIN-closing':crypto_returns['BITCOIN-Close']})
    ## Get Regressors
    crypto_data_returns = crypto_returns.drop('BITCOIN-Close', axis=1)  

    #-------------#
    # Subset Data #
    #-------------#
    # Get indices
    N_train_step = int(round(crypto_data_returns.shape[0]*Train_step_proportion,0))
    N_test_set = int(crypto_data_returns.shape[0] - round(crypto_data_returns.shape[0]*Train_step_proportion,0))
    # # Get Datasets
    X_train = crypto_data_returns[:N_train_step]
    X_test = crypto_data_returns[-N_test_set:]

    ## Coerce into format used in benchmark model(s)
    data_x = X_train
    data_x_test = X_test
    # Get Targets 
    data_y = crypto_target_data[:N_train_step]
    data_y_test = crypto_target_data[-N_test_set:]

    # Scale Data
    scaler = StandardScaler()
    data_x = scaler.fit_transform(data_x)
    data_x_test = scaler.transform(data_x_test)

    # # Update User
    print('#================================================#')
    print(' Training Datasize: '+str(X_train.shape[0])+' and test datasize: ' + str(X_test.shape[0]) + '.  ')
    print('#================================================#')

    # # Set First Run to Off
    First_run = False

    #-----------#
    # Plot Data #
    #-----------#
    fig = crypto_data_returns.plot(figsize=(16, 16))
    fig.get_legend().remove()
    plt.title("Crypto_Market Returns")

    # SAVE Figure to .eps
    plt.savefig('./outputs/plotsANDfigures/Crypto_Data_returns.pdf', format='pdf')
    
    
    # Set option to SnP to port rest of pre-processing automatically that way
    Option_Function = "SnP"


# In[ ]:


# ICML Coercsion
y_train = np.array(data_y).reshape(-1)
y_test = np.array(data_y_test).reshape(-1)

