#!/usr/bin/env python
# coding: utf-8

# # Expert Architope
# ---
# 
# **What this code does:** *This Code Implements the Expert-provided partition version of the architope.*
# 
# **Why is it separate?**  *It can be run in parallel with the other codes since all seeds and states are the same.*

# #### Mode: Code-Testin Parameter(s)

# In[119]:


trial_run = True


# ### Meta-parameters

# In[120]:


# Test-size Ratio
test_size_ratio = 0.3
min_height = 50


# ### Hyperparameters
# 
# Only turn of if running code directly here, typically this script should be run be called by other notebooks.  

# In[121]:


# load dataset
results_path = "./outputs/models/"
results_tables_path = "./outputs/results/"
raw_data_path_folder = "./inputs/raw/"
data_path_folder = "./inputs/data/"


# ### Import

# In[122]:


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

# In[239]:


# Time-Elapse (Start) for Training on Each Part
Architope_partition_training_begin_expt = time.time()
# Initialize running max for Parallel time
Architope_partitioning_max_time_running_exp = -math.inf # Initialize slowest-time at - infinity to force updating!
# Initialize N_parameter counter for Architope
N_params_Architope_expt = 0


# #### Partition Dataset using Expert Opinion

# In[240]:


# Write Column Names #
#--------------------#
# Read from X
X_col_names = (X.drop(['median_house_value'],axis=1).columns).values
# Write to X_train and X_test
X_train.columns= X_col_names
X_test.columns= X_col_names


# In[241]:


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

# In[242]:


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

# In[243]:


# Initialize Ratios
N_ratios = np.array([])

# Build Ratios
for Number_part_i in range(len(X_train_list_expt)):
    # Update Ratios
    N_ratios = np.append(N_ratios,((X_train_list_expt[Number_part_i].shape[0])/X_train.shape[0]))


# #### Build Export Architope

# In[244]:


y_hat_train = np.array([])
y_hat_test = np.array([])
y_train_target_reordered = np.array([])
y_test_target_reordered = np.array([])


# In[245]:


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
                                                                                     X_test_partial=X_train_list_expt[current_part],
                                                                                     X_test=X_test_list_expt[current_part])
    # Update Predictions and Ordering of Targets
    y_hat_train = np.append(y_hat_train,y_hat_train_full_loop)
    y_train_target_reordered = np.append(y_train_target_reordered,(y_train_list_expt[current_part]))
    y_hat_test = np.append(y_hat_test,y_hat_test_full_loop)
    y_test_target_reordered = np.append(y_test_target_reordered,(y_test_list_expt[current_part]))

        
    #============#
    # Timer(end) #
    #============#
    current_part_training_time_for_parallel = time.time() - current_part_training_time_for_parallel_begin
    Architope_partitioning_max_time_running_exp = max(Architope_partitioning_max_time_running_exp,current_part_training_time_for_parallel)

    #============---===============#
    # N_parameter Counter (Update) #
    #------------===---------------#
    N_params_Architope_expt = N_params_Architope_expt + N_params_Architope_loop

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


# In[246]:


# Time-Elapsed Training on Each Part
Architope_partition_training_expt = time.time() - Architope_partition_training_begin_expt


# ---

# #### Write Predictions

# Compute Performance

# In[252]:


# Compute Peformance
performance_Architope_exp = reporter(y_train_hat_in=y_hat_train,
                                    y_test_hat_in=y_hat_test,
                                    y_train_in=y_train_target_reordered,
                                    y_test_in=y_test_target_reordered)
# Write Performance
performance_Architope_exp.to_latex((results_tables_path+"Architopes_expert_performance.tex"))

# Update User
print(performance_Architope_exp)


# ---

# ---

# #### Compute Required Training Time(s)

# ### Model Complexity/Efficiency Metrics

# In[254]:


# Compute Parameters for composite models #
#-----------------------------------------#

# Build AIC-like Metric #
#-----------------------#
AIC_like = 2*(N_params_Architope_expt - np.log((performance_Architope_exp['test']['MAE'])))
AIC_like = np.round(AIC_like,3)
Efficiency = np.log(N_params_Architope_expt) *(performance_Architope_exp['test']['MAE'])
Efficiency = np.round(Efficiency,3)


# Build Table #
#-------------#
Architope_Model_Complexity_Expert = pd.DataFrame({'L-time': [Architope_partition_training_expt],
                                                  'P-time':[Architope_partitioning_max_time_running_exp],
                                                  'N_params_expt': [N_params_Architope_expt],
                                                  'AIC-like': [AIC_like],
                                                  'Eff': [Efficiency]})


# Write Required Training Time(s)
Architope_Model_Complexity_Expert.to_latex((results_tables_path+"ArchitopeExpert_model_complexities.tex"))

#--------------======---------------#
# Display Required Training Time(s) #
#--------------======---------------#
print(Architope_Model_Complexity_Expert)


# ---

# # Summary

# In[260]:


print(' ')
print('#===============#')
print('# Model Summary #')
print('#===============#')
print(' ')
print('------------------------------------')
print('Model Performance: Expert Architope')
print('------------------------------------')
print(performance_Architope_exp)
print(' ')
print('------------------------------------')
print('Model Complexity: Expert Architope')
print('------------------------------------')
print(Architope_Model_Complexity_Expert)
print(' ')
print(' ')
print('😃😃 Have a wonderful day!! 😃😃')


# ---
# # Fin
# ---

# ---

# ---

# ---
