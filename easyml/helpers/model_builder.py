import json

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

    target_df = target_df.values.ravel()

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

    elif algorithm_type_num == ALGORITHM.DECISION_TREE_REGRESSOR:
        model = create_decision_tree_regressor(input_df, target_df, parameters)

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
        save_model(model, alg_type, algorithm_type_num, file_id, parameters)


def save_model(model, alg_type, algorithm_type_num, file_id, parameters):
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
    model_obj.parameters = json.dumps(parameters)
    model_obj.parent_file = CsvFile.objects.get(id=file_id)
    model_obj.save()


def create_linear_regression_model(input_df, target_df, parameters):
    fit_intercept = bool(parameters.get('linreg_fit_intercept', False))
    normalize = bool(parameters.get('linreg_normalize', False))

    lin_reg = LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
    lin_reg = lin_reg.fit(input_df, target_df)

    return lin_reg


def create_logistic_regression_model(input_df, target_df, parameters):
    logreg_penalty = parameters.get('logreg_penalty', 'l2')
    logreg_c_select = parameters.get('logreg_C_select', 'custom')
    logreg_fit_intercept = bool(parameters.get('logreg_fit_intercept', False))

    if logreg_c_select == 'custom':
        logreg_c = int(parameters.get('logreg_C', 1.0))
        logreg = LogisticRegression(C=logreg_c,
                                    penalty=logreg_penalty,
                                    fit_intercept=logreg_fit_intercept,
                                    solver='lbfgs')

    else:
        steps = [('std_scaler', StandardScaler())]
        steps += [('log_regression', LogisticRegression(penalty=logreg_penalty,
                                                        multi_class='auto',
                                                        fit_intercept=logreg_fit_intercept,
                                                        solver='lbfgs'))]
        pipe = Pipeline(steps)

        parameters = {'log_regression__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        logreg = GridSearchCV(estimator=pipe, param_grid=parameters, cv=5)

    clf = logreg.fit(input_df, target_df)
    return clf


def create_linear_discriminant_analysis(input_df, target_df, parameters):
    solver = parameters.get('lda_solver', 'svd')
    clf = LinearDiscriminantAnalysis(solver=solver)
    clf.fit(input_df, target_df)

    return clf


def create_decision_tree_regressor(input_df, target_df, parameters):
    criterion = parameters.get('dtr_criterion', 'mse')
    presort = bool(parameters.get('dtr_presort', False))
    max_depth_choice = parameters.get('dtr_max_depth', 'none')

    if max_depth_choice == 'none':
        best_depth = None

    elif max_depth_choice == 'custom':
        best_depth = parameters.get('dtr_custom_depth', None)

    else:
        x_train, x_valid, y_train, y_valid = train_test_split(input_df, target_df, test_size=0.20)
        r2_lst = []
        depth_iter = 5
        depth_start = 10

        depth_lst = []
        for i in range(depth_iter):
            depth_lst.append(depth_start**i)

        # Select model with best r^2 and least depth
        for depth in depth_lst:
            dt_regr = DecisionTreeRegressor(max_depth=depth, presort=presort, criterion=criterion)
            dt_regr.fit(x_train, y_train)
            r2_lst.append(dt_regr.score(x_valid, y_valid))

        depth_index, r2 = min(enumerate(r2_lst), key=lambda x: abs(x[1] - 1))
        best_depth = depth_lst[depth_index]

    dt_regr_final = DecisionTreeRegressor(max_depth=best_depth,
                                          presort=presort,
                                          criterion=criterion).fit(input_df, target_df)
    return dt_regr_final


def create_gaussian_naive_bayes(input_df, target_df, parameters):
    gnb = GaussianNB()
    gnb.fit(input_df, target_df)

    return gnb


def create_random_forest_classifier(input_df, target_df, parameters):
    criterion = parameters.get('rfc_criterion', 'gini')
    n_estimators = int(parameters.get('rfc_n_estimators', 100))
    depth_select = parameters.get('rfc_max_depth', 'none')

    if depth_select == 'none':
        best_depth = None

    elif depth_select == 'custom':
        best_depth = parameters.get('rfc_custom_depth', None)

    else:
        x_train, x_valid, y_train, y_valid = train_test_split(input_df, target_df, test_size=0.20)

        r2_lst = []
        depth_iter = 5
        depth_start = 10

        depth_lst = []
        for i in range(depth_iter):
            depth_lst.append(depth_start**i)

        # Select model with best r^2 and least depth
        for depth in depth_lst:
            rf_clf = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=depth,
                                            criterion=criterion,
                                            oob_score=True)
            rf_clf.fit(x_train, y_train.values.ravel())
            r2_lst.append(rf_clf.oob_score_)

        depth_index, r2 = min(enumerate(r2_lst), key=lambda x: abs(x[1] - 1))
        best_depth = depth_lst[depth_index]

    rf_clf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=best_depth,
                                    criterion=criterion).fit(input_df, target_df)
    return rf_clf


def create_random_forest_regressor(input_df, target_df, parameters):
    criterion = parameters.get('rfc_criterion', 'mse')
    n_estimators = int(parameters.get('rfc_n_estimators', 100))
    depth_select = parameters.get('rfc_max_depth', 'none')

    if depth_select == 'none':
        best_depth = None

    elif depth_select == 'custom':
        best_depth = parameters.get('rfc_custom_depth', None)

    else:
        x_train, x_valid, y_train, y_valid = train_test_split(input_df, target_df, test_size=0.20)

        r2_lst = []
        depth_iter = 5
        depth_start = 10

        depth_lst = []
        for i in range(depth_iter):
            depth_lst.append(depth_start**i)

        # Select model with best r^2 and least depth
        for depth in depth_lst:
            rf_clf = RandomForestRegressor(n_estimators=n_estimators,
                                           max_depth=depth,
                                           criterion=criterion,
                                           oob_score=True)
            rf_clf.fit(x_train, y_train)
            r2_lst.append(rf_clf.oob_score_)

        depth_index, r2 = min(enumerate(r2_lst), key=lambda x: abs(x[1] - 1))
        best_depth = depth_lst[depth_index]

    rf_clf = RandomForestRegressor(n_estimators=n_estimators,
                                   max_depth=best_depth,
                                   criterion=criterion).fit(input_df, target_df)
    return rf_clf


def create_k_nearest_neighbors_classifier(input_df, target_df, parameters):
    n_neighbors = int(parameters.get('nnc_k', 5))
    weights = parameters.get('weights', 'uniform')
    algorithm = parameters.get('algorithm', 'auto')
    p = int(parameters.get('nnc_p', 2))

    neighbors = KNeighborsClassifier(n_neighbors=n_neighbors,
                                     algorithm=algorithm,
                                     weights=weights,
                                     p=p)
    neighbors.fit(input_df, target_df)

    return neighbors


def create_k_nearest_neighbors_regressor(input_df, target_df, parameters):
    n_neighbors = int(parameters.get('nnc_k', 5))
    weights = parameters.get('weights', 'uniform')
    algorithm = parameters.get('algorithm', 'auto')
    p = int(parameters.get('nnc_p', 2))

    neighbors = KNeighborsRegressor(n_neighbors=n_neighbors,
                                    algorithm=algorithm,
                                    weights=weights,
                                    p=p)
    neighbors.fit(input_df, target_df)

    return neighbors


def create_nearest_centroid(input_df, target_df, parameters):
    clf = NearestCentroid()
    clf.fit(input_df, target_df)

    return clf


def create_support_vector_machine_classifier(input_df, target_df, parameters):
    kernel = parameters.get('svc_kernel', 'rbf')
    degree = int(parameters.get('svc_degree', 3))
    c = parameters.get('svc_C', 1.0)

    clf = svm.SVC(kernel=kernel, degree=degree, C=c)
    clf.fit(input_df, target_df)

    return clf


def create_support_vector_machine_regressor(input_df, target_df, parameters):
    kernel = parameters.get('svr_kernel', 'rbf')
    degree = int(parameters.get('svr_degree', 3))

    clf = svm.SVR(kernel=kernel, degree=degree)
    clf.fit(input_df, target_df)

    return clf

