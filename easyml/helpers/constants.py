class COLUMN_TYPE:
    IGNORE = 0
    INPUT = 1
    TARGET = 2

class ALGORITHM:
    LINEAR_REGRESSION = 1
    LOGISTIC_REGRESSION = 2
    LINEAR_DISCRIMINANT_ANALYSIS = 3
    DECISION_TREE = 4
    GAUSSIAN_NAIVE_BAYES = 5
    RANDOM_FOREST_CLASSIFIER = 6
    RANDOM_FOREST_REGRESSOR = 7
    K_NEAREST_NEIGHBORS = 8
    SUPPORT_VECTOR_MACHINES = 9
    NEAREST_CENTROID = 10


algorithm_name_map = {
    str(ALGORITHM.LINEAR_REGRESSION): 'Linear Regression',
    str(ALGORITHM.LOGISTIC_REGRESSION): 'Logistic Regression',
    str(ALGORITHM.LINEAR_DISCRIMINANT_ANALYSIS): 'Linear Discriminant Analysis',
    str(ALGORITHM.DECISION_TREE): 'Decision Tree',
    str(ALGORITHM.GAUSSIAN_NAIVE_BAYES): 'Gaussian Naive Bayes',
    str(ALGORITHM.RANDOM_FOREST_CLASSIFIER): 'Random Forest Classifier',
    str(ALGORITHM.K_NEAREST_NEIGHBORS): 'K Nearest Neighbors',
    str(ALGORITHM.SUPPORT_VECTOR_MACHINES): 'Support Vector Machines',
    str(ALGORITHM.NEAREST_CENTROID): 'Nearest Centroid',
    str(ALGORITHM.RANDOM_FOREST_REGRESSOR): 'Random Forest Regressor',
}