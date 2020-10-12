trial_run = False
# This one is with larger height

# This file contains the hyper-parameter grids used to train the imprinted-tree nets.

#----------------------#
########################
# Hyperparameter Grids #
########################
#----------------------#

Random_Depths_Readout = 10
Random_Depths = np.array([2])#, 100, 500])#, 3,10, 20, 50, 100, 150,200, 500])
N_Features = 1# Random_Depths.shape[0]
N_Features_Search_Space_Dimension = 10**4


# Hyperparameter Grid (Readout)
#------------------------------#
if trial_run == True:

    
    # Training Parameters
    #----------------------#
    # Number of Jobs (Cores to use)
    n_jobs = 4
    # Number of Random CV Draws
    n_iter = 1
    n_iter_trees = 1#20
    # Number of CV Folds
    CV_folds = 4

    
    # Model Parameters
    #------------------#
    param_grid_Vanilla_Nets = {'batch_size': [8],
                    'epochs': [2],
                      'learning_rate': [0.14],
                      'height': [2],
                       'depth': [1],
                      'input_dim':[15],
                       'output_dim':[1]}
    # Random Forest Grid
    #--------------------#
    Rand_Forest_Grid = {'learning_rate': [0.1],
                        'max_depth': [2],
                        'min_samples_leaf': [1],
                       'n_estimators': [10],
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
                          'input_dim':[15],
                           'output_dim':[1]}
    # Random Forest Grid
    #--------------------#
    Rand_Forest_Grid = {'learning_rate': [0.0001,0.0005,0.005, 0.01],
                        'max_depth': [1,2,3,4,5, 7, 10, 25, 50, 75,100, 150, 200, 300, 500],
                        'min_samples_leaf': [1,2,3,4, 5, 9, 17, 20,50,75, 100],
                       'n_estimators': [5, 10, 25, 50, 100, 200, 250]
                       }
                       

