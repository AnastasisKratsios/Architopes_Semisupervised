# This file contains the hyper-parameter grids used to train the imprinted-tree nets.

# Training Parameters
#----------------------#
# Number of Jobs (Cores to use)
n_jobs = 4
# Number of Random CV Draws
n_iter = 1
n_iter_trees = 1#20
# Number of CV Folds
CV_folds = 4

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
    param_grid_Vanilla_Nets = {'batch_size': [128],
                    'epochs': [800],
                      'learning_rate': [0.146],
                      'height': [200],
                       'depth': [2],
                      'input_dim':[15],
                       'output_dim':[1]}
else:
    param_grid_Vanilla_Nets = {'batch_size': [16,32,64],
                        'epochs': [200, 400, 800, 1000, 1200, 1400],
                          'learning_rate': [0.0001,0.0005,0.005],
                          'height': [100,200, 400],
                           'depth': [1,2,3],
                          'input_dim':[15],
                           'output_dim':[1]}
                       
# Random Forest Grid
#--------------------#
Rand_Forest_Grid = {'learning_rate': [0.1],#, 0.05, 0.02, 0.01, 0.01],
                    'max_depth': [4],#, 6, 10,50, 100],
                    'min_samples_leaf': [3],#, 5, 9, 17, 50, 100, 150, 200],
                   'n_estimators': [1],#00, 200, 500, 1000]
                   }
