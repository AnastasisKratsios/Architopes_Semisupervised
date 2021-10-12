# Load Packages/Modules
exec(open('Init_Dump.py').read())
# Load Hyper-parameter Grid
exec(open('Grid_Enhanced_Network.py').read())
# Import Time (for some reason must be done internally?)
import time

# Initialize time-elapsed computation
Gradient_boosted_Random_forest_time_Begin = time.time()
# Run Random Forest Util
rand_forest_model_grad_boosted = GradientBoostingRegressor()
# Grid-Search CV
Random_Forest_GridSearch = RandomizedSearchCV(estimator = rand_forest_model_grad_boosted,
                   n_iter=n_iter_trees,
                   cv=KFold(CV_folds, random_state=2020, shuffle=True),
                   param_distributions=Rand_Forest_Grid,
                   return_train_score=True,
                   random_state=2020,
                   verbose=10,
                   n_jobs=n_jobs)
random_forest_trained = Random_Forest_GridSearch.fit(X_train,y_train)
random_forest_trained = random_forest_trained.best_estimator_
#--------------------------------------------------#
# Write: Model, Results, and Best Hyper-Parameters #
#--------------------------------------------------#
# Save Readings
cur_path = os.path.expanduser('./outputs/tables/best_params_Gradient_Boosted_Tree.txt')
with open(cur_path, "w") as f:
    f.write(str(Random_Forest_GridSearch.best_params_))

best_params_table_tree = pd.DataFrame({'N Estimators': [Random_Forest_GridSearch.best_params_['n_estimators']],
                                    'Min Samples Leaf': [Random_Forest_GridSearch.best_params_['min_samples_leaf']],
                                    'Learning Rate': [Random_Forest_GridSearch.best_params_['learning_rate']],
                                    'Max Depth': [Random_Forest_GridSearch.best_params_['max_depth']],
                                    })
best_params_table_tree.to_latex('./outputs/tables/Best_params_table_Gradient_Boosted_Tree.txt')
# Update User
print("Gradient Boosted Trees Model - Done CV!")
y_train_hat_random_forest_Gradient_boosting = random_forest_trained.predict(X_train)
y_test_hat_random_forest_Gradient_boosting = random_forest_trained.predict(X_test)
# Evaluate time-elapsed computation
Gradient_boosted_Random_forest_time = time.time() - Gradient_boosted_Random_forest_time_Begin
# Count Number of Parameters in Random Forest Regressor
N_tot_params_per_tree = [ (x[0].tree_.node_count)*random_forest_trained.n_features_ for x in random_forest_trained.estimators_]
N_tot_params_in_forest = sum(N_tot_params_per_tree)
#---------------------------------------------#
# Note to potential code-reader:
# Thanks Philip Casgain for the counter! :D
#---------------------------------------------#
print("Random Forest Regressor uses: "+str(N_tot_params_in_forest)+" parameters.")
# Compute Peformance
Gradient_boosted_tree = reporter(y_train_hat_in=y_train_hat_random_forest_Gradient_boosting,
                                    y_test_hat_in=y_test_hat_random_forest_Gradient_boosting,
                                    y_train_in=y_train,
                                    y_test_in=y_test)
# Write Performance
Gradient_boosted_tree.to_latex((results_tables_path+"Gradient_Boosted_Tree_Performance.tex"))
# Update User
print(Gradient_boosted_tree)
# ---
# # Fin
# ---