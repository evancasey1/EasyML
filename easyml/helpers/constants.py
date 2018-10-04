class COLUMN_TYPE:
    IGNORE = 0
    INPUT = 1
    TARGET = 2

class ALGORITHM:
    LINEAR_REGRESSION = 1
    LOGISTIC_REGRESSION = 2
    LINEAR_DISCRIMINANT_ANALYSIS = 3
    DECISION_TREE = 4
    NAIVE_BAYES = 5
    RANDOM_FOREST = 6
    K_NEAREST_NEIGHBORS = 7
    SUPPORT_VECTOR_MACHINES = 8
    NEAREST_CENTROID = 9


algorithm_name_map = {
    str(ALGORITHM.LINEAR_REGRESSION): 'Linear Regression',
    str(ALGORITHM.LOGISTIC_REGRESSION): 'Logistic Regression',
    str(ALGORITHM.LINEAR_DISCRIMINANT_ANALYSIS): 'Linear Discriminant Analysis',
    str(ALGORITHM.DECISION_TREE): 'Decision Tree',
    str(ALGORITHM.NAIVE_BAYES): 'Naive Bayes',
    str(ALGORITHM.RANDOM_FOREST): 'Random Forest',
    str(ALGORITHM.K_NEAREST_NEIGHBORS): 'K Nearest Neighbors',
    str(ALGORITHM.SUPPORT_VECTOR_MACHINES): 'Support Vector Machines',
    str(ALGORITHM.NEAREST_CENTROID): 'Nearest Centroid',
}