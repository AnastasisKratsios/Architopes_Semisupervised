#!/usr/bin/env python
# coding: utf-8

# # Helper Function(s)
# A little list of useful helper functions when building the architope!

# In[ ]:


# MAPE, between 0 and 100
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    y_true.shape = (y_true.shape[0], 1)
    y_pred.shape = (y_pred.shape[0], 1)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# # Deep Learning Helper(s)

# ## Custom Layers
#  - Fully Conneted Dense: Typical Feed-Forward Layer
#  - Fully Connected Dense Invertible: Necessarily satisfies for input and output layer(s)
#  - Fully Connected Dense Destructor: Violates Assumptions for both input and ouput layer(s) (it is neither injective nor surjective)

# In[ ]:


class fullyConnected_Dense(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(fullyConnected_Dense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='Weights_ffNN',
                                 shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
        self.b = self.add_weight(name='bias_ffNN',
                                 shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
class fullyConnected_Dense_Invertible(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(fullyConnected_Dense_Invertible, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='Weights_ffNN',
                                 shape=(input_shape[-1], input_shape[-1]),
                               initializer='zeros',
                               trainable=True)
        self.b = self.add_weight(name='bias_ffNN',
                                 shape=(self.units,),
                               initializer='zeros',
                               trainable=True)

    def call(self, inputs):
        expw = tf.linalg.expm(self.w)
        return tf.matmul(inputs, expw) + self.b


# In[ ]:


#------------------------------------------------------------------------------------------------#
#                                      Define Predictive Model                                   #
#------------------------------------------------------------------------------------------------#

def def_trainable_layers_Nice_Input_Output(height, depth, learning_rate, input_dim, output_dim):
    #----------------------------#
    # Maximally Interacting Layer #
    #-----------------------------#
    input_layer = tf.keras.Input(shape=(input_dim,))
    
    
    #------------------#
    # Deep Feature Map #
    #------------------#
    # For this implementation we do not use a "deep feature map!"
#     if Depth_Feature_Map >0:
#         for i_feature_depth in range(Depth_Feature_Map):
#             # First Layer
#             if i_feature_depth == 0:
#                 deep_feature_map = fullyConnected_Dense_Invertible(input_dim)(input_layer)
#                 deep_feature_map = tf.nn.leaky_relu(deep_feature_map)
#             else:
#                 deep_feature_map = fullyConnected_Dense_Invertible(input_dim)(deep_feature_map)
#                 deep_feature_map = tf.nn.leaky_relu(deep_feature_map)
#     else:
#         deep_feature_map = input_layer
        
    
    #------------------#
    #   Core Layers    #
    #------------------#
    core_layers = fullyConnected_Dense(height)(input_layer)
    # Activation
    core_layers = tf.nn.swish(core_layers)
    # Train additional Depth?
    if depth>1:
        # Add additional deep layer(s)
        for depth_i in range(1,depth):
            core_layers = fullyConnected_Dense(height)(core_layers)
            # Activation
            core_layers = tf.nn.swish(core_layers)
    
    #------------------#
    #  Readout Layers  #
    #------------------# 
    # Affine (Readout) Layer (Dense Fully Connected)
    output_layers = fullyConnected_Dense(output_dim)(core_layers)  
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layers)
    
    
    #----------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    trainable_layers_model.compile(optimizer=opt, loss="mae", metrics=["mse", "mae", "mape"])

    return trainable_layers_model

#------------------------------------------------------------------------------------------------#
#                                      Build Predictive Model                                    #
#------------------------------------------------------------------------------------------------#

def build_ffNN(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train, X_test_partial,X_test,NOCV=False):    
    if NOCV == False:
        # Deep Feature Network
        Nice_Model_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=def_trainable_layers_Nice_Input_Output, verbose=True)

        # Randomized CV
        Nice_Model_CVer = RandomizedSearchCV(estimator=Nice_Model_CV,
                                             n_jobs=n_jobs,
                                             cv=KFold(n_folds, random_state=2020, shuffle=True),
                                             param_distributions=param_grid_in,
                                             n_iter=n_iter,
                                             return_train_score=True,
                                             random_state=2020,
                                             verbose=10)
    else:
        Nice_Model_CVer = def_trainable_layers_Nice_Input_Output(height = param_grid_in['height'][0],
                                                                 depth = param_grid_in['depth'][0],
                                                                 learning_rate = param_grid_in['learning_rate'][0],
                                                                 input_dim = param_grid_in['input_dim'][0],
                                                                 output_dim = param_grid_in['output_dim'][0])
    
    # Fit Model #
    #-----------#
    if NOCV == False:
        Nice_Model_CVer.fit(X_train,y_train)
    else:
        Nice_Model_CVer.fit(X_train,y_train,epochs = param_grid_in['epochs'][0])

    # Write Predictions #
    #-------------------#
    y_hat_train = Nice_Model_CVer.predict(X_test_partial)
    y_hat_test = Nice_Model_CVer.predict(X_test)
    
    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    if NOCV == False:
        best_model = Nice_Model_CVer.best_estimator_
        # Count Number of Parameters
        N_params_best_ffNN = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    
    # Return Values #
    #---------------#
    if NOCV == False:
        return y_hat_train, y_hat_test, N_params_best_ffNN
    else:
        N_neurons_used = (param_grid_in['depth'][0])*(param_grid_in['height'][0])
        return y_hat_train, y_hat_test, N_neurons_used

# Update User
#-------------#
print('Deep Feature Builder - Ready')


## Randomized Version
from scipy import sparse
from scipy.sparse import random as randsp
from scipy.sparse import csr_matrix
# from sklearn import linear_model
from sklearn.linear_model import LinearRegression
def build_ffNN_random(X_train_in,X_train_in_full,X_test_in,y_train_in,param_grid_in):
    # Initializations
    ## Model
#     clf = linear_model.Lasso(alpha=0.1)
    clf = LinearRegression()
    ## Features to Randomize
    X_train_rand_features = X_train_in.to_numpy()
    X_train_full_rand_features = X_train_in_full.to_numpy()
    X_test_rand_features = X_test_in.to_numpy()
    N_Random_Features = param_grid_in['height'][0]
    N_Random_Features_Depth = param_grid_in['depth'][0]
    for depth in range(N_Random_Features_Depth):
        # Get Random Features
        #---------------------------------------------------------------------------------------------------#
        Weights_rand = randsp(m=(X_train_rand_features.shape[1]),n=N_Random_Features,density = 0.75)
        biases_rand = np.random.uniform(low=-.5,high=.5,size = N_Random_Features)
        ### Apply Random (hidden) Weights
        X_train_rand_features = sparse.csr_matrix.dot(X_train_rand_features,Weights_rand)
        X_train_full_rand_features = sparse.csr_matrix.dot(X_train_full_rand_features,Weights_rand)
        #### Apply Random (hidden) Biases
        X_train_rand_features = X_train_rand_features + biases_rand
        X_train_rand_features = np.sin(X_train_rand_features)
        #####
        X_train_full_rand_features = X_train_full_rand_features + biases_rand
        X_train_full_rand_features = np.sin(X_train_full_rand_features)
        #### Compress
        X_train_rand_features = sparse.csr_matrix(X_train_rand_features)
        X_train_full_rand_features = sparse.csr_matrix(X_train_full_rand_features)

        #------# Test #-------------#
        #### Apply Random (hidden) Weights
        X_test_rand_features = sparse.csr_matrix.dot(X_test_rand_features,Weights_rand) 
        #### Apply Random (hidden) Biases
        X_test_rand_features = X_test_rand_features + biases_rand
        #### Apply Discontinuous (Step) Activation function
        X_test_rand_features = np.sin(X_test_rand_features)
        #### Compress
        X_test_rand_features = sparse.csr_matrix(X_test_rand_features)
    # Get regressor 
    clf.fit(X_train_rand_features,y_train_in)
    y_hat_train = clf.predict(X_train_full_rand_features)
    ### Predict
    y_hat_test = clf.predict(X_test_rand_features)
    
    # Count Parameters
    N_parameters = (N_Random_Features * N_Random_Features_Depth) + N_Random_Features*(param_grid_in['input_dim'][0]+param_grid_in['output_dim'][0])
    # Return Computations
    y_hat_train = y_hat_train.reshape(-1,param_grid_in['output_dim'][0])
    y_hat_test = y_hat_test.reshape(-1,param_grid_in['output_dim'][0])
    return y_hat_train, y_hat_test, N_parameters


#------------------------------------------------------------------------------------------------#
#                                      Define Predictive Model                                   #
#------------------------------------------------------------------------------------------------#
# Keras
# def def_simple_deep_classifer(height, depth, learning_rate, input_dim, output_dim):
#     # Initialize Simple Deep Classifier
#     simple_deep_classifier = tf.keras.Sequential()
#     for d_i in range(depth):
#         simple_deep_classifier.add(tf.keras.layers.Dense(height, activation='relu'))

#     simple_deep_classifier.add(tf.keras.layers.Dense(output_dim, activation='sigmoid'))

#     # Compile Simple Deep Classifier
#     simple_deep_classifier.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
    
#     # Return Output
#     return simple_deep_classifier
# Tensorflow
def def_simple_deep_classifer(height, depth, learning_rate, input_dim, output_dim):
    input_layer = tf.keras.Input(shape=(input_dim,))
    #------------------#
    #   Core Layers    #
    #------------------#
    core_layers = fullyConnected_Dense(height)(input_layer)
    # Activation
    core_layers = tf.nn.swish(core_layers)
    # Train additional Depth?
    if depth>1:
        # Add additional deep layer(s)
        for depth_i in range(1,depth):
            core_layers = fullyConnected_Dense(height)(core_layers)
            # Activation
            core_layers = tf.nn.swish(core_layers)
    
    #------------------#
    #  Readout Layers  #
    #------------------# 
    # Affine (Readout) Layer (Dense Fully Connected)
    core_layers = fullyConnected_Dense(output_dim)(core_layers)
    output_layers = tf.sigmoid(core_layers)
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layers)
    
    
    #----------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    trainable_layers_model.compile(optimizer=opt, loss="mae", metrics=["mse", "mae", "mape"])

    return trainable_layers_model
#------------------------------------------------------------------------------------------------#
#                                  Build Deep Classifier Model                                   #
#------------------------------------------------------------------------------------------------#
from tensorflow.keras import Sequential
def build_simple_deep_classifier(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train, X_test, NOCV=False):
    
    if NOCV == True:
        # Deep Feature Network
        CV_simple_deep_classifier = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=def_simple_deep_classifer, verbose=True)

        # Randomized CV
        CV_simple_deep_classifier_CVer = RandomizedSearchCV(estimator=CV_simple_deep_classifier, 
                                        n_jobs=n_jobs,
                                        cv=KFold(n_folds, random_state=2020, shuffle=True),
                                        param_distributions=param_grid_in,
                                        n_iter=n_iter,
                                        return_train_score=True,
                                        random_state=2020,
                                        verbose=10)
    else:
        CV_simple_deep_classifier_CVer = def_simple_deep_classifer(height = param_grid_in['height'][0],
                                                                   depth = param_grid_in['depth'][0],
                                                                   learning_rate = param_grid_in['learning_rate'][0],
                                                                   input_dim = param_grid_in['input_dim'][0],
                                                                   output_dim = param_grid_in['output_dim'][0])
    
    # Fit Model #
    #-----------#
    if NOCV == False:
        CV_simple_deep_classifier_CVer.fit(X_train,y_train)
    else:
        CV_simple_deep_classifier_CVer.fit(X_train,
                                           y_train,
                                           epochs = param_grid_in['epochs'][0])

    # Make Prediction(s)
    predicted_classes_train = CV_simple_deep_classifier_CVer.predict(X_train)
    predicted_classes_test = CV_simple_deep_classifier_CVer.predict(X_test)
    
    # Counter number of parameters #
    #------------------------------#
    if NOCV == True:
        # Extract Best Model
        best_model = CV_simple_deep_classifier_CVer.best_estimator_
        # Count Number of Parameters
        N_params_best_classifier = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    
    
    # Return Values #
    #---------------#
    if NOCV == True:
        return predicted_classes_train, predicted_classes_test, N_params_best_classifier
    else:
        return predicted_classes_train, predicted_classes_test, 0
# Update User
#-------------#
print('Deep Classifier - Ready')


#------------------------------------------------------------------------------------------------#
#                            Build Deep Classifier via Randomized Approach                       #
#------------------------------------------------------------------------------------------------#

def build_deep_classifier_random(X_train_in,
                                 X_train_in_full,
                                 X_test_in,
                                 predictions_test_in,
                                 predictions_train_in,
                                 classes_in,
                                 param_grid_in):
    ###----------------------------------------------------###
    print('Generating Random Deep Features for Deep Zero-Sets')
    ## Features to Randomize
    if isinstance(X_train_in, pd.DataFrame): #Coercsion
        X_train_in = X_train_in.to_numpy()
        X_train_in_full = X_train_in_full.to_numpy()
        X_test_in = X_test_in.to_numpy()
    X_train_rand_features = X_train_in
    X_train_full_rand_features = X_train_in_full
    X_test_rand_features = X_test_in
    N_Random_Features = param_grid_in['height'][0]
    N_Random_Features_Depth = param_grid_in['depth'][0]
    for depth in range(N_Random_Features_Depth):
        # Get Random Features
        #---------------------------------------------------------------------------------------------------#
        Weights_rand = randsp(m=(X_train_rand_features.shape[1]),n=N_Random_Features,density = 0.75)
        biases_rand = np.random.uniform(low=-.5,high=.5,size = N_Random_Features)
        ### Apply Random (hidden) Weights
        X_train_rand_features = sparse.csr_matrix.dot(X_train_rand_features,Weights_rand)
        X_train_full_rand_features = sparse.csr_matrix.dot(X_train_full_rand_features,Weights_rand)
        #### Apply Random (hidden) Biases
        X_train_rand_features = X_train_rand_features + biases_rand
        X_train_rand_features = np.sin(X_train_rand_features)
        #####
        X_train_full_rand_features = X_train_full_rand_features + biases_rand
        X_train_full_rand_features = np.sin(X_train_full_rand_features)
        #### Compress
        X_train_rand_features = sparse.csr_matrix(X_train_rand_features)
        X_train_full_rand_features = sparse.csr_matrix(X_train_full_rand_features)

        #------# Test #-------------#
        #### Apply Random (hidden) Weights
        X_test_rand_features = sparse.csr_matrix.dot(X_test_rand_features,Weights_rand) 
        #### Apply Random (hidden) Biases
        X_test_rand_features = X_test_rand_features + biases_rand
        #### Apply Discontinuous (Step) Activation function
        X_test_rand_features = np.sin(X_test_rand_features)
        #### Compress
        X_test_rand_features = sparse.csr_matrix(X_test_rand_features)
#     if isinstance(X_train_rand_features, np.ndarray): #Coercsion
    X_train_rand_features = X_train_rand_features.toarray()
#     if isinstance(X_train_full_rand_features, np.ndarray): #Coercsion
    X_train_full_rand_features = X_train_full_rand_features.toarray()
#     if isinstance(X_test_rand_features, np.ndarray): #Coercsion
    X_test_rand_features = X_test_rand_features.toarray()
    ## Add Skip-Connection
    print('Added Skip Connections')
    X_train_rand_features = np.concatenate((X_train_in,X_train_rand_features),axis=1)
    X_train_full_rand_features = np.concatenate((X_train_in_full,X_train_full_rand_features),axis=1)
    X_test_rand_features = np.concatenate((X_test,X_test_rand_features),axis=1)
    print(X_test_rand_features.shape)
    print('Get Classifier')
    # Initialize Classifier
    parameters = {'penalty': ['none','l2'], 'C': [0.1, 0.5, 1.0, 10, 100, 1000]}
    lr = LogisticRegression(random_state=2020)
    cv = RepeatedStratifiedKFold(n_splits = 4, n_repeats=n_iter, random_state=0)
    classifier = RandomizedSearchCV(lr, parameters, random_state=2020)
    # Train Logistic Classifier #
    #---------------------------#
    warnings.simplefilter("ignore") # Supress warnings
    classifier.fit(X_train_rand_features, classes_in) # Fit Grid-Searched Classifiers
    # Training Set
    predicted_classes_train = classifier.best_estimator_.predict(X_train_full_rand_features).reshape(-1,1)
    PCNN_prediction_y_train = np.take_along_axis(predictions_train_in, predicted_classes_train, axis=1)
    # Testing Set
    predicted_classes_test = classifier.best_estimator_.predict(X_test_rand_features).reshape(-1,1)
    PCNN_prediction_y_test = np.take_along_axis(predictions_test_in, predicted_classes_test, axis=1)
    # Extract Number of Parameters Logistic Regressor
    N_params_deep_Classifier = (classifier.best_estimator_.coef_.shape[0])*(classifier.best_estimator_.coef_.shape[1]) + len(classifier.best_estimator_.intercept_)

    # Count Parameters
    N_parameters = N_params_deep_Classifier + N_Random_Features*(param_grid_in['input_dim'][0]+param_grid_in['output_dim'][0])
    # Get Outputs
    return PCNN_prediction_y_train, PCNN_prediction_y_test, N_parameters

#-------------------------------#
#=### Results & Summarizing ###=#
#-------------------------------#
def reporter(y_train_hat_in,y_test_hat_in,y_train_in,y_test_in):
    # Training Performance
    Training_performance = np.array([mean_absolute_error(y_train_hat_in,y_train_in),
                                mean_squared_error(y_train_hat_in,y_train_in),
                                   mean_absolute_percentage_error(y_train_hat_in,y_train_in)])
    # Testing Performance
    Test_performance = np.array([mean_absolute_error(y_test_hat_in,y_test_in),
                                mean_squared_error(y_test_hat_in,y_test_in),
                                   mean_absolute_percentage_error(y_test_hat_in,y_test_in)])
    # Organize into Dataframe
    Performance_dataframe = pd.DataFrame({'train': Training_performance,'test': Test_performance})
    Performance_dataframe.index = ["MAE","MSE","MAPE"]
    # return output
    return Performance_dataframe


# Other Functions
def softminn(x):
    softmin_output = np.exp(-x) / np.sum(np.exp(-x), axis=0)
    return softmin_output