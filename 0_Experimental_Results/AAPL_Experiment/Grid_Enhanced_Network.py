# Which financial dataset do you want to consider (NB this meta-parameter does not impact the non-financial architopes module)
# Options: SnP or crypto
Option_Function = "AAPL"

# Is this a trial run (to test hardware?)
trial_run = False
# This one is with larger height

# This file contains the hyper-parameter grids used to train the imprinted-tree nets.

#----------------------#
########################
# Hyperparameter Grids #
########################
#----------------------#


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
    CV_folds = 2

    
    # Model Parameters
    #------------------#
    param_grid_Vanilla_Nets = {'batch_size': [16],
                    'epochs': [200],
                      'learning_rate': [0.0014],
                      'height': [50],
                       'depth': [2],
                      'input_dim':[15],
                       'output_dim':[1]}

    param_grid_Deep_Classifier = {'batch_size': [16],
                        'epochs': [50],
                        'learning_rate': [0.01],
                        'height': [4],
                        'depth': [2],
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
                        'epochs': [200, 300, 400, 500, 600],
                          'learning_rate': [0.00001,0.00005,0.0001,0.0005],
                          'height': [50,100, 150, 200],
                           'depth': [2,3],
                          'input_dim':[15],
                           'output_dim':[1]}

    param_grid_Deep_Classifier = {'batch_size': [16,32,64],
                        'epochs': [50, 100, 200, 300],
                        'learning_rate': [0.00001,0.00005,0.0001, 0.0005, 0.001, 0.005],
                        'height': [50,100,150],
                        'depth': [2,3],
                        'input_dim':[15],
                        'output_dim':[1]}
                           
    # Random Forest Grid
    #--------------------#
    Rand_Forest_Grid = {'learning_rate': [0.0001,0.0005,0.005, 0.01],
                        'max_depth': [1,2,3,4,5, 7, 10, 25, 50, 75,100, 150, 200, 300, 500],
                        'min_samples_leaf': [1,2,3,4, 5, 9, 17, 20,50,75, 100],
                       'n_estimators': [5, 10, 25, 50, 100, 200, 250]
                       }
                       

