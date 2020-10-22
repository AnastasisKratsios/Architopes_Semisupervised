#!/usr/bin/env python
# coding: utf-8

# # Expert Architope: Semi-Supervised
# ---
# 
# **What this code does:** *This Code Implements the Expert-provided partition version of the architope.  Then, it uses it to (accelarate the learning of) learn the optimal model-dependant partition $K_{\cdot}^{\star}$.*
# 
# **Why is it separate?**  *It can be run in parallel with the other codes since all seeds and states are the same.*

# #### Mode: Code-Testin Parameter(s)

# In[1]:


trial_run = True


# ### Meta-parameters

# In[2]:


# Test-size Ratio
test_size_ratio = 0.3
min_height = 50


# ### Hyperparameters
# 
# Only turn of if running code directly here, typically this script should be run be called by other notebooks.  

# In[3]:


# load dataset
results_path = "./outputs/models/"
results_tables_path = "./outputs/results/"
raw_data_path_folder = "./inputs/raw/"
data_path_folder = "./inputs/data/"


# ### Import

# In[4]:


# Load Packages/Modules
exec(open('Init_Dump.py').read())
# Load Hyper-parameter Grid
exec(open('Grid_Enhanced_Network.py').read())
# Load Helper Function(s)
exec(open('Helper_Functions.py').read())
# Pre-process Data
exec(open('Prepare_Data_California_Housing.py').read())
# Import time separately
import time


# #### Begin Times and Other Trackers

# In[5]:


# Time-Elapse (Start) for Training on Each Part
Architope_partition_training_begin_expt = time.time()
# Initialize running max for Parallel time
Architope_partitioning_max_time_running_exp = -math.inf # Initialize slowest-time at - infinity to force updating!
# Initialize N_parameter counter for Architope
N_params_Architope_deep_expt = 0


# #### Partition Dataset using Expert Opinion

# In[6]:


# Write Column Names #
#--------------------#
# Read from X
X_col_names = (X.drop(['median_house_value'],axis=1).columns).values
# Write to X_train and X_test
X_train.columns= X_col_names
X_test.columns= X_col_names


# In[7]:


# Initialize Lists
#------------------#
## Initialize Data Lists
X_train_list_expt = list()
y_train_list_expt = list()
X_test_list_expt = list()
y_test_list_expt = list()

## Initialize Predictions Lists
y_hat_train_list_expt = list()
y_hat_test_list_expt = list()


# #### Define Parts (expert opinion) and append to lists

# In[8]:


#----------#
# 1H Ocean #
#----------#

# Extract Train
X_train_list_expt.append(X_train[X_train['ocean_proximity_<1H OCEAN']==1])
y_train_list_expt.append(y_train[X_train['ocean_proximity_<1H OCEAN']==1])
# Extract Test
X_test_list_expt.append(X_test[X_test['ocean_proximity_<1H OCEAN']==1])
y_test_list_expt.append(y_test[X_test['ocean_proximity_<1H OCEAN']==1])

#--------#
# INLAND #
#--------#

# Extract Train
X_train_list_expt.append(X_train[X_train['ocean_proximity_INLAND']==1])
y_train_list_expt.append(y_train[X_train['ocean_proximity_INLAND']==1])
# Extract Test
X_test_list_expt.append(X_test[X_test['ocean_proximity_INLAND']==1])
y_test_list_expt.append(y_test[X_test['ocean_proximity_INLAND']==1])



#----------#
# NEAR BAY #
#----------#

# Extract Train
X_train_list_expt.append(X_train[X_train['ocean_proximity_NEAR BAY']==1])
y_train_list_expt.append(y_train[X_train['ocean_proximity_NEAR BAY']==1])
# Extract Test
X_test_list_expt.append(X_test[X_test['ocean_proximity_NEAR BAY']==1])
y_test_list_expt.append(y_test[X_test['ocean_proximity_NEAR BAY']==1])


#------------------------#
# NEAR OCEAN & on ISLAND #
#------------------------#

# Extract Train
X_train_list_expt.append(X_train[np.logical_or(X_train['ocean_proximity_ISLAND']==1,(X_train['ocean_proximity_NEAR OCEAN']==1))])
y_train_list_expt.append(y_train[np.logical_or(X_train['ocean_proximity_ISLAND']==1,(X_train['ocean_proximity_NEAR OCEAN']==1))])
# Extract Test
X_test_list_expt.append(X_test[np.logical_or(X_test['ocean_proximity_ISLAND']==1,(X_test['ocean_proximity_NEAR OCEAN']==1))])
y_test_list_expt.append(y_test[np.logical_or(X_test['ocean_proximity_ISLAND']==1,(X_test['ocean_proximity_NEAR OCEAN']==1))])


# #### Build Ratios

# In[9]:


# Initialize Ratios
N_ratios = np.array([])

# Build Ratios
for Number_part_i in range(len(X_train_list_expt)):
    # Update Ratios
    N_ratios = np.append(N_ratios,((X_train_list_expt[Number_part_i].shape[0])/X_train.shape[0]))


# Initialize Timers and Benchmarking tool(s)

# In[10]:


# Time-Elapse (Start) for Training on Each Part
Architope_partition_training_begin = time.time()
# Initialize running max for Parallel time
Architope_partitioning_max_time_running = -math.inf # Initialize slowest-time at - infinity to force updating!


# #### Build Export Architope

# In[11]:


for current_part in range(len(X_train_list_expt)):
    #==============#
    # Timer(begin) #
    #==============#
    current_part_training_time_for_parallel_begin = time.time()
    
    
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
    print('Status: Current part: ' + str(current_part) + ' out of : '+str(len(X_train_list_expt)) +' parts.')
    print('Heights to iterate over: '+str(current_height))
    
    # Generate Prediction(s) on current Part #
    #----------------------------------------#
    # Failsafe (number of data-points)
    CV_folds_failsafe = min(CV_folds,max(1,(X_train.shape[0]-1)))
    # Train Network
    y_hat_train_full_loop, y_hat_test_full_loop, N_params_Architope_loop = build_ffNN(n_folds = CV_folds_failsafe, 
                                                                                     n_jobs = n_jobs,
                                                                                     n_iter = n_iter, 
                                                                                     param_grid_in = param_grid_Vanilla_Nets, 
                                                                                     X_train= X_train_list_expt[current_part], 
                                                                                     y_train=y_train_list_expt[current_part],
                                                                                     X_test_partial=X_train,
                                                                                     X_test=X_test)
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
        
    #============#
    # Timer(end) #
    #============#
    current_part_training_time_for_parallel = time.time() - current_part_training_time_for_parallel_begin
    Architope_partitioning_max_time_running_exp = max(Architope_partitioning_max_time_running_exp,current_part_training_time_for_parallel)

    #============---===============#
    # N_parameter Counter (Update) #
    #------------===---------------#
    N_params_Architope_deep_expt = N_params_Architope_deep_expt + N_params_Architope_loop

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


# In[12]:


# Time-Elapsed Training on Each Part
Architope_partition_training_expt = time.time() - Architope_partition_training_begin_expt


# ---

# ## Phase 2) Train Deep Classifier with Expert "Prior" Partition $K_{\cdot}^{exp}$!

# #### Time-Elapsed Training Deep Classifier

# In[13]:


Architope_deep_classifier_training_begin_deep_exp = time.time()


# #### Initialize Classes (aka targets for classification task)

# In[14]:


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

# In[15]:


# Re-Load Hyper-parameter Grid
exec(open('Grid_Enhanced_Network.py').read())
# Re-Load Helper Function(s)
exec(open('Helper_Functions.py').read())

# Redefine (Dimension-related) Elements of Grid
param_grid_Deep_Classifier['input_dim'] = [X_train.shape[1]]
param_grid_Deep_Classifier['output_dim'] = [partition_labels_training.shape[1]]


# #### Train Classifier

# In[16]:


# Train simple deep classifier
predicted_classes_train, predicted_classes_test, N_params_deep_classifier = build_simple_deep_classifier(n_folds = CV_folds, 
                                                                                                        n_jobs = n_jobs, 
                                                                                                        n_iter =n_iter, 
                                                                                                        param_grid_in=param_grid_Deep_Classifier, 
                                                                                                        X_train = X_train, 
                                                                                                        y_train = partition_labels_training,
                                                                                                        X_test = X_test)


# #### Time-Elapsed Training Deep Classifier

# In[17]:


Architope_deep_classifier_training_deep_exp = time.time() - Architope_deep_classifier_training_begin_deep_exp


# #### Make Prediction(s)

# In[18]:


# Training Set
Architope_prediction_y_train_deep_exp = np.take_along_axis(predictions_train, predicted_classes_train[:,None], axis=1)
# Testing Set
Architope_prediction_y_test_deep_exp = np.take_along_axis(predictions_test, predicted_classes_test[:,None], axis=1)


# #### Write Predictions

# Compute Performance

# In[19]:


# Compute Peformance
performance_Architope_deep_exp = reporter(y_train_hat_in=Architope_prediction_y_train_deep_exp,
                                    y_test_hat_in=Architope_prediction_y_test_deep_exp,
                                    y_train_in=y_train,
                                    y_test_in=y_test)
# Write Performance
performance_Architope_deep_exp.to_latex((results_tables_path+"Architopes_deep_expert_performance.tex"))

# Update User
print(performance_Architope_deep_exp)


# ---

# #### Compute Required Training Time(s)

# ### Model Complexity/Efficiency Metrics

# In[20]:


# Compute Parameters for composite models #
#-----------------------------------------#
# Compute training times
Architope_Deep_expert_L_Time = Architope_deep_classifier_training_deep_exp + Architope_partition_training_expt
Architope_Deep_expert_p_Time = Architope_deep_classifier_training_deep_exp + Architope_partitioning_max_time_running_exp

# Build AIC-like Metric #
#-----------------------#
AIC_like = 2*(N_params_Architope_deep_expt - np.log((performance_Architope_deep_exp['test']['MAE'])))
AIC_like = np.round(AIC_like,3)
Efficiency = np.log(N_params_Architope_deep_expt) *(performance_Architope_deep_exp['test']['MAE'])
Efficiency = np.round(Efficiency,3)


# Build Table #
#-------------#
Architope_Model_Complexity_Deep_Expert = pd.DataFrame({'L-time': [Architope_partition_training_expt],
                                                  'P-time':[Architope_partitioning_max_time_running_exp],
                                                  'N_params_expt': [N_params_Architope_deep_expt],
                                                  'AIC-like': [AIC_like],
                                                  'Eff': [Efficiency]})


# Write Required Training Time(s)
Architope_Model_Complexity_Deep_Expert.to_latex((results_tables_path+"Architope_DEEP_Expert_model_complexities.tex"))

#--------------======---------------#
# Display Required Training Time(s) #
#--------------======---------------#
print(Architope_Model_Complexity_Deep_Expert)


# ---

# # Summary

# In[21]:


print(' ')
print('#===============#')
print('# Model Summary #')
print('#===============#')
print(' ')
print('------------------------------------')
print('Model Performance: Expert Architope')
print('------------------------------------')
print(performance_Architope_deep_exp)
print(' ')
print('------------------------------------')
print('Model Complexity: Expert Architope')
print('------------------------------------')
print(Architope_Model_Complexity_Deep_Expert)
print(' ')
print(' ')
print('😃😃 Have a wonderful day!! 😃😃')


# ---
# # Fin
# ---
