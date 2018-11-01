from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from .constants import COLUMN_TYPE, ALGORITHM, ALGORITHM_NAME_MAP
from mainsite.models import CsvFile, CsvFileData, MLModel
from .util import get_dataframe


def create_model(algorithm_type_num, file_id, parameters):
    file_data = CsvFileData.objects.filter(parent_file_id=file_id)\
        .exclude(type=COLUMN_TYPE.IGNORE).order_by('column_num')

    if file_data.count() == 0:
        print("Error: No data for file {}".format(file_id))
        return

    input_data = file_data.filter(type=COLUMN_TYPE.INPUT).order_by('column_num')
    target_data = file_data.filter(type=COLUMN_TYPE.TARGET)

    model = None
    alg_type = ALGORITHM_NAME_MAP[str(algorithm_type_num)]

    input_df = get_dataframe(input_data)
    target_df = get_dataframe(target_data)

    if algorithm_type_num == ALGORITHM.LINEAR_REGRESSION:
        model = create_linear_regression_model(input_df, target_df, parameters)

    elif algorithm_type_num == ALGORITHM.K_NEAREST_NEIGHBORS_CLASSIFIER:
        model = create_k_nearest_neighbors_classifier(input_df, target_df, parameters)

    elif algorithm_type_num == ALGORITHM.K_NEAREST_NEIGHBORS_REGRESSOR:
        model = create_k_nearest_neighbors_regressor(input_df, target_df, parameters)

    elif algorithm_type_num == ALGORITHM.LOGISTIC_REGRESSION:
        model = create_logistic_regression_model(input_df, target_df, parameters)

    elif algorithm_type_num == ALGORITHM.NEAREST_CENTROID:
        model = create_nearest_centroid(input_df, target_df, parameters)

    elif algorithm_type_num == ALGORITHM.LINEAR_DISCRIMINANT_ANALYSIS:
        model = create_linear_discriminant_analysis(input_df, target_df, parameters)

    elif algorithm_type_num == ALGORITHM.DECISION_TREE:
        model = create_decision_tree(input_df, target_df, parameters)

    elif algorithm_type_num == ALGORITHM.GAUSSIAN_NAIVE_BAYES:
        model = create_gaussian_naive_bayes(input_df, target_df, parameters)

    elif algorithm_type_num == ALGORITHM.RANDOM_FOREST_CLASSIFIER:
        model = create_random_forest_classifier(input_df, target_df, parameters)

    elif algorithm_type_num == ALGORITHM.RANDOM_FOREST_REGRESSOR:
        model = create_random_forest_regressor(input_df, target_df, parameters)

    elif algorithm_type_num == ALGORITHM.SUPPORT_VECTOR_MACHINE_CLASSIFIER:
        model = create_support_vector_machine_classifier(input_df, target_df, parameters)

    elif algorithm_type_num == ALGORITHM.SUPPORT_VECTOR_MACHINE_REGRESSOR:
        model = create_support_vector_machine_regressor(input_df, target_df, parameters)

    if model:
        save_model(model, alg_type, algorithm_type_num, file_id)

def save_model(model, alg_type, algorithm_type_num, file_id):
    parent_file = CsvFile.objects.get(id=file_id)
    display_name = "{}: {}".format(parent_file.display_name, alg_type)

    same_name_count = MLModel.objects.filter(name=parent_file.display_name, type=alg_type).count()
    if same_name_count > 0:
        display_name += ' ({})'.format(same_name_count)

    model_obj = MLModel()
    model_obj.type = alg_type
    model_obj.type_num = algorithm_type_num
    model_obj.data = model
    model_obj.name = parent_file.display_name
    model_obj.display_name = display_name
    model_obj.parent_file = CsvFile.objects.get(id=file_id)
    model_obj.save()

def create_linear_regression_model(input_df, target_df, parameters):
    lin_reg = LinearRegression().fit(input_df, target_df)

    return lin_reg

def create_logistic_regression_model(input_df, target_df, parameters):
    steps = [('std_scaler', StandardScaler())]
    steps += [('log_regression', LogisticRegression(penalty='l2', multi_class='auto'))]
    pipe = Pipeline(steps)

    parameters = {'log_regression__C': [0.001, 0.01, 0.1, 1, 10, 100]}
    gs = GridSearchCV(estimator=pipe, param_grid=parameters, cv=5)

    clf = gs.fit(input_df, target_df)
    return clf

def create_linear_discriminant_analysis(input_df, target_df, parameters):
    clf = LinearDiscriminantAnalysis()
    clf.fit(input_df, target_df)

    return clf

def create_decision_tree(input_df, target_df, parameters):
    x_train, x_valid, y_train, y_valid = train_test_split(input_df, target_df, test_size=0.20)
    r2_lst = []
    depth_mul = 10
    depth_iter = 10
    depth_start = 1.0 / (depth_mul ** (depth_iter / 2))

    depth_lst = []
    for i in range(depth_iter):
        depth_lst.append(depth_start * (depth_mul ** i))

    # Select model with best r^2 and least depth
    for depth in depth_lst:
        dt_regr = DecisionTreeRegressor(max_depth=depth)
        dt_regr.fit(x_train, y_train)
        r2_lst.append(dt_regr.score(x_valid, y_valid))

    depth_index, r2 = min(enumerate(r2_lst), key=lambda x: abs(x[1] - 1))
    best_depth = depth_lst[depth_index]

    dt_regr_final = DecisionTreeRegressor(max_depth=best_depth).fit(input_df, target_df)
    return dt_regr_final

def create_gaussian_naive_bayes(input_df, target_df, parameters):
    gnb = GaussianNB()
    gnb.fit(input_df, target_df)

    return gnb

def create_random_forest_classifier(input_df, target_df, parameters):
    x_train, x_valid, y_train, y_valid = train_test_split(input_df, target_df, test_size=0.20)

    n_est = 100
    r2_lst = []
    depth_mul = 10
    depth_iter = 10
    depth_start = 1.0/(depth_mul**(depth_iter/2))

    depth_lst = []
    for i in range(depth_iter):
        depth_lst.append(depth_start * (depth_mul**i))

    # Select model with best r^2 and least depth
    for depth in depth_lst:
        rf_clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, oob_score=True)
        rf_clf.fit(x_train, y_train.values.ravel())
        r2_lst.append(rf_clf.oob_score_)

    depth_index, r2 = min(enumerate(r2_lst), key=lambda x: abs(x[1] - 1))
    best_depth = depth_lst[depth_index]

    rf_clf = RandomForestClassifier(n_estimators=n_est, max_depth=best_depth).fit(input_df, target_df.values.ravel())
    return rf_clf

def create_random_forest_regressor(input_df, target_df, parameters):
    x_train, x_valid, y_train, y_valid = train_test_split(input_df, target_df, test_size=0.20)

    n_est = 100
    r2_lst = []
    depth_mul = 10
    depth_iter = 10
    depth_start = 1.0 / (depth_mul ** (depth_iter / 2))

    depth_lst = []
    for i in range(depth_iter):
        depth_lst.append(depth_start * (depth_mul ** i))

    # Select model with best r^2 and least depth
    for depth in depth_lst:
        rf_clf = RandomForestRegressor(n_estimators=n_est, max_depth=depth, oob_score=True)
        rf_clf.fit(x_train, y_train.values.ravel())
        r2_lst.append(rf_clf.oob_score_)

    depth_index, r2 = min(enumerate(r2_lst), key=lambda x: abs(x[1] - 1))
    best_depth = depth_lst[depth_index]

    rf_clf = RandomForestRegressor(n_estimators=n_est, max_depth=best_depth).fit(input_df, target_df.values.ravel())
    return rf_clf

def create_k_nearest_neighbors_classifier(input_df, target_df, parameters):
    neighbors = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
    neighbors.fit(input_df, target_df)

    return neighbors

def create_k_nearest_neighbors_regressor(input_df, target_df, parameters):
    neighbors = KNeighborsRegressor(n_neighbors=5, algorithm='auto')
    neighbors.fit(input_df, target_df)

    return neighbors

def create_nearest_centroid(input_df, target_df, parameters):
    clf = NearestCentroid()
    clf.fit(input_df, target_df)

    return clf

def create_support_vector_machine_classifier(input_df, target_df, parameters):
    clf = svm.SVC()
    clf.fit(input_df, target_df)

    return clf

def create_support_vector_machine_regressor(input_df, target_df, parameters):
    clf = svm.SVR()
    clf.fit(input_df, target_df)

    return clf

