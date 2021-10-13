# Which financial dataset do you want to consider (NB this meta-parameter does not impact the non-financial architopes module)
# Options: AAPL, SnP, or crypto (or Motivational_Example for DEMO version!)
Option_Function = "SnP" 

# Is this a trial run (to test hardware?)
trial_run = True
# This one is with larger height

# This file contains the hyper-parameter grids used to train the imprinted-tree nets.

#----------------------#
########################
# Hyperparameter Grids #
########################
#----------------------#
# Test-size Ratio
test_size_ratio = 0.9
min_width = 100
min_epochs = 100; min_epochs_classifier = 100
# Ablation Finess
N_plot_finess = 4
# min_parts_threshold = .001; max_parts_threshold = 0.9
N_min_parts = 1; N_max_plots = 100
Tied_Neurons_Q = True
randomize_subpattern_construction = True
randomize_subpattern_construction_Deep_ZeroSets = True
# Partition with Inputs (determine parts with domain) or outputs (determine parts with image)
Partition_using_Inputs = True
# Cuttoff Level
gamma = .5
# Softmax Layer instead of sigmoid
softmax_layer = False
N_parts_possibilities = np.array([1,2,3,5,20,50,200,400]); N_plot_finess = len(N_parts_possibilities)

# Tables
Relative_MAE_to_FFNN = True

#------------------------------------#
# Only For Motivational Example Only #
#------------------------------------#
## Hyperparameters
percentage_in_row = .2
N = 10**4

# Motivational Example (OLD)
# def f_1(x):
#     return 1 + np.sin(10*x)
# def f_2(x):
#     return -2 -x**2
# x_0 = 0
# x_end = 1
import seaborn as sns

#----------------------------------#
# Only For Synthetic Examples Only #
#----------------------------------#
# In addition to the above
noise_level = .01
tailedness = 30
frequency_or_self_paritioning = .25
def f_unknown(x):
    if x % frequency_or_self_paritioning >= frequency_or_self_paritioning/2:
        y = 1 + np.exp(x)*np.cos(x)
    else:
        y = -1-(x**2)*np.cos(x)
    return y
D_in = 100

# Hyperparameter Grid (Readout)
#------------------------------#
if trial_run == True:

    
    # Training Parameters
    #----------------------#
    # Number of Jobs (Cores to use)
    n_jobs = 4
    # Number of Random CV Draws
    n_iter = 1
    n_iter_trees = 5
    # Number of CV Folds
    CV_folds = 2

    
    # Model Parameters
    #------------------#
    # Training Parameters
    #----------------------#
    # Number of Jobs (Cores to use)
    n_jobs = 4
    # Number of Random CV Draws
    n_iter = 1
    n_iter_trees = 5
    # Number of CV Folds
    CV_folds = 2

    
    # Model Parameters
    #------------------#
    
    # Training Parameters
    #----------------------#
    # Number of Jobs (Cores to use)
    n_jobs = 4
    # Number of Random CV Draws
    n_iter = 1
    n_iter_trees = 1#20
    # Number of CV Folds
    CV_folds = 2

    
    # Model Parameters
    #------------------#
    param_grid_FFNNs = {'batch_size': [16],
                    'epochs': [100],
                      'learning_rate': [0.00001],
                      'height': [200],
                      'depth': [3],
                      'input_dim':[1],
                      'output_dim':[1]}
    param_grid_Vanilla_Nets = {'batch_size': [16],
                    'epochs': [200],
                      'learning_rate': [0.00001],
                      'height': [800],
                      'depth': [2],
                      'input_dim':[1],
                      'output_dim':[1]}

    param_grid_Deep_Classifier = {'batch_size': [16],
                        'epochs': [200],
                        'learning_rate': [0.00001],
                        'height': [800],
                        'depth': [1],
                        'input_dim':[1],
                        'output_dim':[1]}
    
                       
    # Random Forest Grid
    #--------------------#
    Rand_Forest_Grid = {'learning_rate': [0.0001],
                        'max_depth': [2,3],
                        'min_samples_leaf': [5,6,8],
                       'n_estimators': [100],
                       }
    
else:
    
    # Training Parameters
    #----------------------#
    # Number of Jobs (Cores to use)
    n_jobs = 70
    # Number of Random CV Draws
    n_iter = 50
    n_iter_trees = 50
    # Number of CV Folds
    CV_folds = 4
    
    
    # Model Parameters
    #------------------#
    param_grid_Vanilla_Nets = {'batch_size': [16,32,64],
                        'epochs': [200, 400, 800, 1000, 1200, 1400],
                          'learning_rate': [0.0001,0.0005,0.005],
                          'height': [100,200, 400, 600, 800],
                           'depth': [1,2],
                          'input_dim':[1],
                           'output_dim':[1]}

    param_grid_Deep_Classifier = {'batch_size': [16,32,64],
                        'epochs': [200, 400, 800, 1000, 1200, 1400],
                        'learning_rate': [0.0001,0.0005,0.005, 0.01],
                        'height': [100,200, 400, 500,600],
                        'depth': [1,2,3,4],
                        'input_dim':[15],
                        'output_dim':[1]}
                           
    # Random Forest Grid
    #--------------------#
    Rand_Forest_Grid = {'learning_rate': [0.0001,0.0005,0.005, 0.01],
                        'max_depth': [1,2,3,4,5, 7, 10, 25, 50, 75,100, 150, 200, 300, 500],
                        'min_samples_leaf': [1,2,3,4, 5, 9, 17, 20,50,75, 100],
                       'n_estimators': [5, 10, 25, 50, 100, 200, 250]
                       }
                       

