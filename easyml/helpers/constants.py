class COLUMN_TYPE:
    IGNORE = 0
    INPUT = 1
    TARGET = 2

class ALGORITHM:
    LINEAR_REGRESSION = 1
    LOGISTIC_REGRESSION = 2
    LINEAR_DISCRIMINANT_ANALYSIS = 3
    DECISION_TREE_REGRESSOR = 4
    GAUSSIAN_NAIVE_BAYES = 5
    RANDOM_FOREST_CLASSIFIER = 6
    RANDOM_FOREST_REGRESSOR = 7
    K_NEAREST_NEIGHBORS_CLASSIFIER = 8
    K_NEAREST_NEIGHBORS_REGRESSOR = 9
    SUPPORT_VECTOR_MACHINE_CLASSIFIER = 10
    SUPPORT_VECTOR_MACHINE_REGRESSOR = 11
    NEAREST_CENTROID = 12


ALGORITHM_NAME_MAP = {
    str(ALGORITHM.LINEAR_REGRESSION): 'Linear Regression',
    str(ALGORITHM.LOGISTIC_REGRESSION): 'Logistic Regression',
    str(ALGORITHM.LINEAR_DISCRIMINANT_ANALYSIS): 'Linear Discriminant Analysis',
    str(ALGORITHM.DECISION_TREE_REGRESSOR): 'Decision Tree Regressor',
    str(ALGORITHM.GAUSSIAN_NAIVE_BAYES): 'Gaussian Naive Bayes',
    str(ALGORITHM.RANDOM_FOREST_CLASSIFIER): 'Random Forest Classifier',
    str(ALGORITHM.RANDOM_FOREST_REGRESSOR): 'Random Forest Regressor',
    str(ALGORITHM.K_NEAREST_NEIGHBORS_CLASSIFIER): 'K Nearest Neighbors Classifier',
    str(ALGORITHM.K_NEAREST_NEIGHBORS_REGRESSOR): 'K Nearest Neighbors Regressor',
    str(ALGORITHM.SUPPORT_VECTOR_MACHINE_CLASSIFIER): 'Support Vector Machine Classifier',
    str(ALGORITHM.SUPPORT_VECTOR_MACHINE_REGRESSOR): 'Support Vector Machine Regressor',
    str(ALGORITHM.NEAREST_CENTROID): 'Nearest Centroid',
}

ALGORITHM_PARAM_MAP = {
    str(ALGORITHM.LINEAR_REGRESSION): ['linreg_normalize', 'linreg_fit_intercept'],
    str(ALGORITHM.LOGISTIC_REGRESSION): ['logreg_fit_intercept', 'logreg_C', 'logreg_C_select', 'logreg_penalty'],
    str(ALGORITHM.LINEAR_DISCRIMINANT_ANALYSIS): ['lda_solver'],
    str(ALGORITHM.DECISION_TREE_REGRESSOR): ['dtr_criterion', 'dtr_presort', 'dtr_max_depth', 'dtr_custom_depth'],
    str(ALGORITHM.GAUSSIAN_NAIVE_BAYES): [],
    str(ALGORITHM.RANDOM_FOREST_CLASSIFIER): ['rfc_criterion', 'rfc_n_estimators', 'rfc_max_depth', 'rfc_custom_depth'],
    str(ALGORITHM.RANDOM_FOREST_REGRESSOR): ['rfr_criterion', 'rfr_n_estimators', 'rfr_max_depth', 'rfr_custom_depth'],
    str(ALGORITHM.K_NEAREST_NEIGHBORS_CLASSIFIER): ['nnc_weights', 'nnc_algorithm', 'nnc_k', 'nnc_p'],
    str(ALGORITHM.K_NEAREST_NEIGHBORS_REGRESSOR): ['nnr_weights', 'nnr_algorithm', 'nnr_k', 'nnr_p'],
    str(ALGORITHM.SUPPORT_VECTOR_MACHINE_CLASSIFIER): ['svc_degree', 'svc_C'],
    str(ALGORITHM.SUPPORT_VECTOR_MACHINE_REGRESSOR): ['svr_degree'],
    str(ALGORITHM.NEAREST_CENTROID): [],
}
