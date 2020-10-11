#!/usr/bin/env python
# coding: utf-8

# # Random Lipschitz Partition
# 
# We implement the random paritioning method of [Yair Bartal](https://scholar.google.com/citations?user=eCXP24kAAAAJ&hl=en):
# - [On approximating arbitrary metrices by tree metrics](https://dl.acm.org/doi/10.1145/276698.276725)
# 
# The algorithm is summarized as follow:
# 
# ---
# 
# ## Algorithm:
#  1. Sample $\alpha \in [4^{-1},2^{-1}]$ randomly and uniformly,
#  2. Apply a random suffle of the data, (a random bijection $\pi:\{i\}_{i=1}^X \rightarrow \mathbb{X}$),
#  3. For $i = 1,\dots,I$:
#    - Set $K_i\triangleq B\left(\pi(i),\alpha \Delta \right) - \bigcup_{j=1}^{i-1} P_j$
#  
#  4. Remove empty members of $\left\{K_i\right\}_{i=1}^X$.  
#  
#  **Return**: $\left\{K_i\right\}_{i=1}^{\tilde{X}}$.  
#  
#  For more details on the random-Lipschitz partition of Yair Bartal, see this [well-written blog post](https://nickhar.wordpress.com/2012/03/26/lecture-22-random-partitions-of-metric-spaces/).

# ### Meta-parameters

# In[1]:


# Test-size Ratio
test_size_ratio = 0.3
min_height = 100


# ### Hyperparameters
# 
# Only turn of if running code directly here, typically this script should be run be called by other notebooks.  

# In[2]:


# load dataset
results_path = "./outputs/models/"
results_tables_path = "./outputs/results/"
raw_data_path_folder = "./inputs/raw/"
data_path_folder = "./inputs/data/"


# ### Import

# In[ ]:


# Load Packages/Modules
exec(open('Init_Dump.py').read())
# Load Hyper-parameter Grid
exec(open('Grid_Enhanced_Network.py').read())
# Load Helper Function(s)
exec(open('Helper_Functions.py').read())
# Pre-process Data
exec(open('Prepare_Data_California_Housing.py').read())


# #### Pre-Process:
# - Convert Categorical Variables to Dummies
# - Remove Bad Column
# - Perform Training/Test Split

# ### Define Random Partition Builder

# In[ ]:


from scipy.spatial import distance_matrix


# Here we use $\Delta_{in} = Q_{q}\left(\Delta(\mathbb{X})\right)$ where $\Delta(\mathbb{X})$ is the vector of (Euclidean) distances between the given data-points, $q \in (0,1)$ is a hyper-parameter, and $Q$ is the empirical quantile function.

# In[ ]:


def Random_Lipschitz_Partioner(Min_data_size_percentage,q_in, X_train_in,y_train_in, CV_folds_failsafe, min_size):
    #-----------------------#
    # Reset Seed Internally #
    #-----------------------#
    random.seed(2020)
    np.random.seed(2020)

    #-------------------------------------------#
    #-------------------------------------------#
    # 1) Sample radius from unifom distribution #
    #-------------------------------------------#
    #-------------------------------------------#
    alpha = np.random.uniform(low=.25,high=.5,size=1)[0]

    #-------------------------------------#
    #-------------------------------------#
    # 2) Apply Random Bijection (Shuffle) #
    #-------------------------------------#
    #-------------------------------------#
    X_train_in_shuffled = X_train_in#.sample(frac=1)
    y_train_in_shuffled = y_train_in#.sample(frac=1)

    #--------------------#
    #--------------------#
    # X) Initializations #
    #--------------------#
    #--------------------#
    # Compute-data-driven radius
    Delta_X = distance_matrix(X_train_in_shuffled,X_train_in_shuffled)[::,0]
    Delta_in = np.quantile(Delta_X,q_in)

    # Initialize Random Radius
    rand_radius = Delta_in*alpha

    # Initialize Data_sizes & ratios
    N_tot = X_train_in.shape[0] #<- Total number of data-points in input data-set!
    N_radios = np.array([])
    N_pool_train_loop = N_tot
    # Initialize List of Dataframes
    X_internal_train_list = list()
    y_internal_train_list = list()

    # Initialize Partioned Data-pool
    X_internal_train_pool = X_train_in_shuffled
    y_internal_train_pool = y_train_in_shuffled

    # Initialize counter 
    part_current_loop = 0

    #----------------------------#
    #----------------------------#
    # 3) Iteratively Build Parts #
    #----------------------------#
    #----------------------------#

    while ((N_pool_train_loop/N_tot > Min_data_size_percentage) or (X_internal_train_pool.empty == False)):
        # Extract Current Center
        center_loop = X_internal_train_pool.iloc[0]
        # Compute Distances
        ## Training
        distances_pool_loop_train = X_internal_train_pool.sub(center_loop)
        distances_pool_loop_train = np.array(np.sqrt(np.square(distances_pool_loop_train).sum(axis=1)))
        # Evaluate which Distances are less than the given random radius
        Part_train_loop = X_internal_train_pool[distances_pool_loop_train<rand_radius]
        Part_train_loop_y = y_internal_train_pool[distances_pool_loop_train<rand_radius]

        # Remove all data-points which are "too small"
        if X_internal_train_pool.shape[0] > max(CV_folds,4):
            # Append Current part to list
            X_internal_train_list.append(Part_train_loop)
            y_internal_train_list.append(Part_train_loop_y)

        # Remove current part from pool 
        X_internal_train_pool = X_internal_train_pool[(np.logical_not(distances_pool_loop_train<rand_radius))]
        y_internal_train_pool = y_internal_train_pool[(np.logical_not(distances_pool_loop_train<rand_radius))]

        # Update Current size of pool of training data
        N_pool_train_loop = X_internal_train_pool.shape[0]
        N_radios = np.append(N_radios,(N_pool_train_loop/N_tot))

        # Update Counter
        part_current_loop = part_current_loop +1
        
        # Update User
        print((N_pool_train_loop/N_tot))


    # Post processing #
    #-----------------#
    # Remove Empty Partitions
    N_radios = N_radios[N_radios>0]
    
    # Sanity Check #
    #---------------#
    # Check if partitioning makes sense and report minimum partition size
#     sanity_checker = 0
#     min_size_partition = math.inf
#     for i in range(len(X_internal_train_list)):
#         sanity_checker = sanity_checker + X_internal_train_list[i].shape[0]
#         min_size_partition = min(min_size_partition,X_internal_train_list[i].shape[0])
#     if(sanity_checker == X_train_in.shape[0]):
#         print('Status: Everything Adds up...partitioning makes sense!')
#         print('Minimum partition size: '+str(min_size_partition))
#     else:
#         print('Status: Warning something went wrong in partitioning: Probably some points are unnasigned!')
    
    
    #-----------------------------------------------------------------#
    # Combine parts which are too small to perform CV without an error
    #-----------------------------------------------------------------#
    # Initialize lists (partitions) with "enough" datums per part
    X_internal_train_list_good = list()
    y_internal_train_list_good = list()
    # Initialize first list item test
    is_first = True
    # Initialize counter
    goods_counter = 0
    for search_i in range(len(X_internal_train_list)):
        number_of_instances_in_part = len(X_internal_train_list[search_i]) 
        if number_of_instances_in_part < max(CV_folds_failsafe,min_size):
            # Check if first 
            if is_first:
                # Initialize set of small X_parts
                X_small_parts = X_internal_train_list[search_i]
                # Initialize set of small y_parts
                y_small_parts = y_internal_train_list[search_i]

                # Set is_first to false
                is_first = False
            else:
                X_small_parts = X_small_parts.append(X_internal_train_list[search_i])
                y_small_parts = np.append(y_small_parts,y_internal_train_list[search_i])
        else:
            # Append to current list
            X_internal_train_list_good.append(X_internal_train_list[search_i])
            y_internal_train_list_good.append(y_internal_train_list[search_i])
            # Update goods counter 
            goods_counter = goods_counter +1

    # Append final one to good list
    X_internal_train_list_good.append(X_small_parts)
    y_internal_train_list_good.append(y_small_parts)

    # reset is_first to false (inscase we want to re-run this particular block)
    is_first = True

    # Set good lists to regular lists
    X_internal_train_list = X_internal_train_list_good
    y_internal_train_list = y_internal_train_list_good
    
    
    
    # Return Value #
    #--------------#
    return [X_internal_train_list, y_internal_train_list, N_radios]


# # Apply Random Partitioner to the given Dataset

# In[ ]:


X_parts_list, y_parts_list, N_ratios = Random_Lipschitz_Partioner(Min_data_size_percentage=.5, q_in=.8, X_train_in=X_train, y_train_in=y_train, CV_folds_failsafe=CV_folds,min_size = 500)


# In[ ]:


print('The_parts_listhe number of parts are: ' + str(len(X_parts_list))+'.')


# #### Building Training Predictions on each part
# - Train locally (on each "naive part")
# - Generate predictions for (full) training and testings sets respectively, to be used in training the classifer and for prediction, respectively.  
# - Generate predictions on all of testing-set (will be selected between later using classifier)

# In[ ]:


for current_part in range(len(X_parts_list)):
    # Initializations #
    #-----------------#
    # Reload Grid
    exec(open('Grid_Enhanced_Network.py').read())
    # Modify heights according to optimal (data-driven) rule (with threshold)
    current_height = np.ceil(np.array(param_grid_Vanilla_Nets['height'])*N_ratios[current_part])
    current_height_threshold = np.repeat(min_height,(current_height.shape[0]))
    current_height = np.maximum(current_height,current_height_threshold)
    current_height = current_height.astype(int).tolist()
    param_grid_Vanilla_Nets['height'] = current_height
    # Automatically Fix Input Dimension
    param_grid_Vanilla_Nets['input_dim'] = [X_train.shape[1]]
    param_grid_Vanilla_Nets['output_dim'] = [1]
    
    # Update User #
    #-------------#
    print('Status: Current part: ' + str(current_part) + ' out of : '+str(len(X_parts_list)) +' parts.')
    print('Heights to iterate over: '+str(current_height))
    
    # Generate Prediction(s) on current Part #
    #----------------------------------------#
    # Failsafe (number of data-points)
    CV_folds_failsafe = min(CV_folds,max(1,(X_train.shape[0]-1)))
    # Train Network
    y_hat_train_full_loop, y_hat_test_full_loop = build_ffNN(n_folds = CV_folds_failsafe, n_jobs = n_jobs,n_iter = n_iter, param_grid_in = param_grid_Vanilla_Nets, X_train= X_parts_list[current_part], y_train=y_parts_list[current_part],X_test_partial=X_train ,X_test=X_test)
    
    # Append predictions to data-frames
    ## If first prediction we initialize data-frames
    if current_part==0:
        # Register quality
        training_quality = np.array(np.abs(y_hat_train_full_loop-y_train))
        training_quality = training_quality.reshape(training_quality.shape[0],1)

        # Save Predictions
        predictions_train = y_hat_train_full_loop
        predictions_train = predictions_train.reshape(predictions_train.shape[0],1)
        predictions_test = y_hat_test_full_loop
        predictions_test = predictions_test.reshape(predictions_test.shape[0],1)
    ## If not first prediction we append to already initialized dataframes
    else:
    # Register Best Scores
        #----------------------#
        # Write Predictions 
        # Save Predictions
        y_hat_train_loop = y_hat_train_full_loop.reshape(predictions_train.shape[0],1)
        predictions_train = np.append(predictions_train,y_hat_train_loop,axis=1)
        y_hat_test_loop = y_hat_test_full_loop.reshape(predictions_test.shape[0],1)
        predictions_test = np.append(predictions_test,y_hat_test_loop,axis=1)
        
        # Evaluate Errors #
        #-----------------#
        # Training
        prediction_errors = np.abs(y_hat_train_loop.reshape(-1,)-y_train)
        training_quality = np.append(training_quality,prediction_errors.reshape(training_quality.shape[0],1),axis=1)

# Update User
#-------------#
print(' ')
print(' ')
print(' ')
print('----------------------------------------------------')
print('Feature Generation (Learning Phase): Score Generated')
print('----------------------------------------------------')
print(' ')
print(' ')
print(' ')


# ---

# ### Train Classifier

# #### Deep Classifier
# Prepare Labels/Classes

# In[ ]:


# Initialize Classes Labels
partition_labels_training_integers = np.argmin(training_quality,axis=-1)
partition_labels_training = pd.DataFrame(pd.DataFrame(partition_labels_training_integers) == 0)
# Build Classes
for part_column_i in range(1,(training_quality.shape[1])):
    partition_labels_training = pd.concat([partition_labels_training,
                                           (pd.DataFrame(partition_labels_training_integers) == part_column_i)
                                          ],axis=1)
# Convert to integers
partition_labels_training = partition_labels_training+0


# Re-Load Grid and Redefine Relevant Input/Output dimensions in dictionary.

# In[ ]:


# Re-Load Hyper-parameter Grid
exec(open('Grid_Enhanced_Network.py').read())
# Re-Load Helper Function(s)
exec(open('Helper_Functions.py').read())

# Redefine (Dimension-related) Elements of Grid
param_grid_Vanilla_Nets['input_dim'] = [X_train.shape[1]]
param_grid_Vanilla_Nets['output_dim'] = [partition_labels_training.shape[1]]


# #### Train Model

# In[ ]:


# Train simple deep classifier
predicted_classes_train, predicted_classes_test = build_simple_deep_classifier(n_folds = CV_folds, 
                                                                    n_jobs = n_jobs, 
                                                                    n_iter =n_iter, 
                                                                    param_grid_in=param_grid_Vanilla_Nets, 
                                                                    X_train = X_train, 
                                                                    y_train = partition_labels_training,
                                                                    X_test = X_test)


# #### Write Predictions

# In[ ]:


# Training Set
Architope_prediction_y_train = np.take_along_axis(predictions_train, predicted_classes_train[:,None], axis=1)
# Testing Set
Architope_prediction_y_test = np.take_along_axis(predictions_test, predicted_classes_test[:,None], axis=1)

# Compute Performance
train_performance = np.array([mean_absolute_error(Architope_prediction_y_train,y_train),mean_squared_error(Architope_prediction_y_train,y_train),mean_absolute_percentage_error(Architope_prediction_y_train,y_train)])
test_performance = np.array([mean_absolute_error(Architope_prediction_y_test,y_test),mean_squared_error(Architope_prediction_y_test,y_test),mean_absolute_percentage_error(Architope_prediction_y_test,y_test)])
# Compile Performance Metrics
performance_Architope = pd.DataFrame({'train': train_performance,'test': test_performance})
performance_Architope.index = ["MAE","MSE","MAPE"]

# Write Performance
performance_Architope.to_latex((results_tables_path+"Architopes_full_performance.tex"))

# Update User
print(performance_Architope)


# ---

# ---

# # Benchmarks

# ---

# ---

# ### Architope with Logistic-Classifier Partitioning

# #### Train Logistic Classifier (Benchmark)

# In[ ]:


parameters = {'penalty': ['none','l1', 'l2'], 'C': [0.1, 0.5, 1.0, 10, 100, 1000]}
lr = LogisticRegression(random_state=2020)
cv = RepeatedStratifiedKFold(n_splits=CV_folds, n_repeats=n_iter, random_state=0)
classifier = RandomizedSearchCV(lr, parameters, random_state=2020)

# Initialize Classes Labels
partition_labels_training = np.argmin(training_quality,axis=-1)


# #### Train Logistic Classifier

# In[ ]:


# Update User #
#-------------#
print("Training classifier and generating partition!")

# Train Logistic Classifier #
#---------------------------#
# Supress warnings caused by "ignoring C" for 'none' penalty and similar obvious warnings
warnings.simplefilter("ignore")
# Train Classifier
classifier.fit(X_train, partition_labels_training)


# #### Write Predicted Class(es)

# In[ ]:


# Training Set
predicted_classes_train_logistic_BM = classifier.best_estimator_.predict(X_train)
Architope_prediction_y_train_logistic_BM = np.take_along_axis(predictions_train, predicted_classes_train_logistic_BM[:,None], axis=1)

# Testing Set
predicted_classes_test_logistic_BM = classifier.best_estimator_.predict(X_test)
Architope_prediction_y_test_logistic_BM = np.take_along_axis(predictions_test, predicted_classes_test_logistic_BM[:,None], axis=1)


# #### Compute Performance

# In[ ]:


# Compute Performance
train_performance_logistic_BM = np.array([mean_absolute_error(Architope_prediction_y_train_logistic_BM,y_train),
                                          mean_squared_error(Architope_prediction_y_train_logistic_BM,y_train),
                                          mean_absolute_percentage_error(Architope_prediction_y_train_logistic_BM,y_train)])
test_performance_logistic_BM = np.array([mean_absolute_error(Architope_prediction_y_test_logistic_BM,y_test),
                                         mean_squared_error(Architope_prediction_y_test_logistic_BM,y_test),
                                         mean_absolute_percentage_error(Architope_prediction_y_test_logistic_BM,y_test)])
# Compile Performance Metrics
performance_logistic_BM = pd.DataFrame({'train': train_performance_logistic_BM,'test': test_performance_logistic_BM})
performance_logistic_BM.index = ["MAE","MSE","MAPE"]

# Write Performance
performance_logistic_BM.to_latex((results_tables_path+"Architopes_logistic_performance.tex"))

# Update User
print(performance_logistic_BM)


# ---

# ## Bagged Feed-Forward Networks (ffNNs)

# In[ ]:


# Train Bagging Weights in-sample
bagging_coefficients = LinearRegression().fit(predictions_train,y_train)

# Predict Bagging Weights out-of-sample
bagged_prediction_train = bagging_coefficients.predict(predictions_train)
bagged_prediction_test = bagging_coefficients.predict(predictions_test)


# In[ ]:


# Compute Performance
train_performance_bag = np.array([mean_absolute_error(bagged_prediction_train,y_train),
                                  mean_squared_error(bagged_prediction_train,y_train),
                                  mean_absolute_percentage_error(bagged_prediction_train,y_train)])
test_performance_bag = np.array([mean_absolute_error(bagged_prediction_test,y_test),
                                 mean_squared_error(bagged_prediction_test,y_test),
                                 mean_absolute_percentage_error(bagged_prediction_test,y_test)])
# Compile Performance Metrics
performance_bagged = pd.DataFrame({'train': train_performance_bag,'test': test_performance_bag})
performance_bagged.index = ["MAE","MSE","MAPE"]

# Write Performance
performance_bagged.to_latex((results_tables_path+"ffNN_Bagged.tex"))

# Update User
print("Written Bagged Performance")
print(performance_bagged)


# In[ ]:


print("Random Partition: Generated!...Feature Generation Complete!")


# ## Vanilla ffNN

# #### Reload Hyper-parameter Grid

# In[ ]:


# Re-Load Hyper-parameter Grid
exec(open('Grid_Enhanced_Network.py').read())
# Re-Load Helper Function(s)
exec(open('Helper_Functions.py').read())


# In[ ]:


#X_train vanilla ffNNs
y_hat_train_Vanilla_ffNN, y_hat_test_Vanilla_ffNN = build_ffNN(n_folds = CV_folds_failsafe, 
                                                               n_jobs = n_jobs, 
                                                               n_iter = n_iter, 
                                                               param_grid_in = param_grid_Vanilla_Nets, 
                                                               X_train=X_train, 
                                                               y_train=y_train, 
                                                               X_test_partial=X_train,
                                                               X_test=X_test)


# In[ ]:


# Update User #
#-------------#
print("Trained vanilla ffNNs")


# #### Evaluate Performance

# In[ ]:


# Compute Performance
train_performance_Vanilla_ffNN = np.array([mean_absolute_error(y_hat_train_Vanilla_ffNN,y_train),mean_squared_error(y_hat_train_Vanilla_ffNN,y_train),mean_absolute_percentage_error(y_hat_train_Vanilla_ffNN,y_train)])
test_performance_Vanilla_ffNN = np.array([mean_absolute_error(y_hat_test_Vanilla_ffNN,y_test),mean_squared_error(y_hat_test_Vanilla_ffNN,y_test),mean_absolute_percentage_error(y_hat_test_Vanilla_ffNN,y_test)])
# Compile Performance Metrics
performance_Vanilla_ffNN = pd.DataFrame({'train': train_performance_Vanilla_ffNN,'test': test_performance_Vanilla_ffNN})
performance_Vanilla_ffNN.index = ["MAE","MSE","MAPE"]

# Write Performance
performance_Vanilla_ffNN.to_latex((results_tables_path+"ffNN_Vanilla.tex"))

# Update User #
#-------------#
print("Written Bagged Vanilla ffNNs")
print(performance_Vanilla_ffNN)


# # Summary

# In[ ]:


print(performance_Architope)
print(performance_logistic_BM)
print(performance_bagged)
print(performance_Vanilla_ffNN)


# ---
# # Fin
# ---
