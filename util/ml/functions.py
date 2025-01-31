import pandas as pd
import os
import gc as hp_optim_gc
from itertools import product
import optuna
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
from matplotlib import rcParams
from data import DATA, _COMMON_PROPS
from json import dumps as jsonDumps
from util.gui.widgets import InProgressWindow

PLOT_PROPS = DATA['plot_properties']
OPTUNA_TOTAL_TRIALS = _COMMON_PROPS['hp_optim']['optuna_total_trials']

# Set global font properties
rcParams['font.family'] = PLOT_PROPS['RC_PARAMS']['FONT_STYLE']
rcParams['font.size'] = PLOT_PROPS['RC_PARAMS']['FONT_SIZE']
rcParams['font.weight'] = PLOT_PROPS['RC_PARAMS']['FONT_WEIGHT'] 

def CHECK_DIR(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Function to check if both files have identical column sets
def CHECK_XLS_FILES(train_file_path: str, test_file_path: str):
    warnings=[]
    try:
        train_df = pd.read_excel(train_file_path)
    except ImportError as e:
        if 'openpyxl' in str(e):
            warnings.append("Missing dependency: openpyxl is required to read Excel files.")
        else:
            warnings.append(f"Error reading Train file: {e}")
        train_df = None
    except Exception as e:
        warnings.append(f"Error reading Train file: {e}")
        train_df = None
    
    try:
        test_df = pd.read_excel(test_file_path)
    except ImportError as e:
        if 'openpyxl' in str(e):
            warnings.append("Missing dependency: openpyxl is required to read Excel files.")
        else:
            warnings.append(f"Error reading Test file: {e}")
        test_df = None
    except Exception as e:
        warnings.append(f"Error reading Test file: {e}")
        test_df = None

    if train_df is not None and test_df is not None:
        if set(train_df.columns) != set(test_df.columns):
            warnings.append("The files do not have the same columns.")
        else:
            try:
                # Drop the first and last columns
                common_cols = train_df.iloc[:, 1:-1]  
                return True, common_cols
            except Exception as e:
                warnings.append(f"Error filtering columns: {e}")

    if len(warnings):
        return False, []

def HP_OPTIM_GENERATE_RESULTS (hp_optim_methodInstance: BaseSearchCV, hp_optim_method_name: str, scoring: str, featureList: list, trainFilePath: str) -> dict:
    trainDF = pd.read_excel(trainFilePath)
    columnsToSelect = featureList
    x_train = trainDF.loc[:, columnsToSelect]
    y_train = trainDF.iloc[:, -1]
    hp_optim_methodInstance.fit(x_train, y_train)

    xls_out = pd.DataFrame(hp_optim_methodInstance.cv_results_)
    CHECK_DIR('output')
    xls_out.to_excel(
        excel_writer=os.path.join('output', f'{hp_optim_method_name}-{scoring}.xlsx'), 
        index=False
    )
    return hp_optim_methodInstance.best_params_

def MODEL_BUILD_GENERATE_RESULTS (regressor, featureList: list, trainFilePath: str, testFilePath: str, regressorName: str) -> dict:
    trainDF = pd.read_excel(trainFilePath)
    testDF = pd.read_excel(testFilePath)
    columnsToSelect = featureList
    x_train = trainDF.loc[:, columnsToSelect]
    y_train = trainDF.iloc[:, -1]
    x_test = testDF.loc[:, columnsToSelect]
    y_test = testDF.iloc[:,-1]

    # MODEL TRAINING
    regressor.fit(x_train, y_train)
    y_train_pred = regressor.predict(x_train)
    y_test_pred = regressor.predict(x_test)
    y_score1 = regressor.predict_proba(x_train)[:,1]
    y_score2 = regressor.predict_proba(x_test)[:,1]

    # PREPARE RESULTS..
    results = {
        'accuracy_train' : round(accuracy_score(y_train, y_train_pred), 4),
        'accuracy_test' : round(accuracy_score(y_test, y_test_pred), 4),
        'precision_train' : round(precision_score(y_train, y_train_pred), 4),
        'precision_test' : round(precision_score(y_test, y_test_pred), 4),
        'recall_train' : round(recall_score(y_train, y_train_pred), 4),
        'recall_test' : round(recall_score(y_test, y_test_pred), 4),
        'f1_score_train' : round(f1_score(y_train, y_train_pred), 4),
        'f1_score_test' : round(f1_score(y_test, y_test_pred), 4),
        'matthews_corrcoef_train' : round(matthews_corrcoef(y_train, y_train_pred), 4),
        'matthews_corrcoef_test' : round(matthews_corrcoef(y_test, y_test_pred), 4),
        'cohen_kappa_score_train' : round(cohen_kappa_score(y_train, y_train_pred), 4),
        'cohen_kappa_score_test' : round(cohen_kappa_score(y_test, y_test_pred), 4),
        'roc_auc_score_train' : round(roc_auc_score(y_train, y_score1), 4),
        'roc_auc_score_test' : round(roc_auc_score(y_test, y_score2), 4),
    }

    # GENERATE <>.xlsx 
    rfr_pred1 = pd.DataFrame(
        {'Y_train': y_train, 'Y_train_pred': y_train_pred}
    )
    rfr_pred2 = pd.DataFrame(
        {'Y_test': y_test,'Y_test_pred': y_test_pred}
    )
    rfr_results = pd.concat([rfr_pred1, rfr_pred2], axis=1)
    
    CHECK_DIR('output')
    rfr_results.to_excel(
        excel_writer=os.path.join('output', f'{regressorName}.xlsx'), 
        index=False
    )

    # GENERATE <>_Results.txt
    output_list_rfr = [
        'Confusion Matrix of Training set:',
        confusion_matrix(y_train, y_train_pred),
        'Confusion Matrix of Test set:',
        confusion_matrix(y_test, y_test_pred),
        'Classification Report for Training set:',
        classification_report(y_train, y_train_pred),
        'Classification Report for Test set:',
        classification_report(y_test, y_test_pred),
        'parameters:',
        str(regressor.get_params())
    ]

    with open(file=os.path.join('output',f'{regressorName}_Results.txt'), mode='w', encoding='utf-8') as my_file_rfr:
        for output in output_list_rfr:
            my_file_rfr.write(str(output) + '\n')

    # GENERATE <>_RocCurve_Test.png
    roc_te = RocCurveDisplay.from_estimator(regressor, x_test, y_test)
    plt.clf()
    fig,ax = plt.subplots(1, figsize=(10,10))
    roc_te.plot(
        color=PLOT_PROPS['ROC_CURVE']['COLOR'], 
        lw=PLOT_PROPS['ROC_CURVE']['LW']
    )
    plt.plot(
        [0, 1], 
        ls="--", 
        color=PLOT_PROPS['DIAGONAL_REF_LINE']['COLOR']
    )
    plt.title(
        'ROC Curve for Test data', 
        fontsize=PLOT_PROPS['TITLE']['FONT_SIZE'], 
        fontweight=PLOT_PROPS['TITLE']['FONT_WEIGHT'], 
        fontname=PLOT_PROPS['TITLE']['FONT_STYLE']
    )
    plt.savefig(
        os.path.join('output', f'{regressorName}_RocCurve_Test.png')
    )
    plt.close(fig)

    # GENERATE <>_RocCurve_Train.png
    roc_tr = RocCurveDisplay.from_estimator(regressor, x_train, y_train)
    plt.clf()
    fig,ax = plt.subplots(1, figsize=(10,10))
    roc_tr.plot(
        color=PLOT_PROPS['ROC_CURVE']['COLOR'], 
        lw=PLOT_PROPS['ROC_CURVE']['LW']
    )
    plt.plot(
        [0, 1], 
        ls="--", 
        color=PLOT_PROPS['DIAGONAL_REF_LINE']['COLOR']
    )
    plt.title(
        'ROC Curve for Train data', 
        fontsize=PLOT_PROPS['TITLE']['FONT_SIZE'], 
        fontweight=PLOT_PROPS['TITLE']['FONT_WEIGHT'], 
        fontname=PLOT_PROPS['TITLE']['FONT_STYLE']
    )
    plt.savefig(
        os.path.join('output', f'{regressorName}_RocCurve_Train.png')
    )
    plt.close(fig)

    return results

PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM = {'FEATURES','METHOD','SCORING','CROSS_FOLD_VALID'}

def _MAP_PARAMRANGE_TO_ACTUAL_LISTOFVALUES (rangeMap: dict):
    if type(rangeMap['_FROM'])==float and '_STEP' not in rangeMap.keys():
        return np.linspace(rangeMap['_FROM'], rangeMap['_TO'], num=25)
    
    if '_STEP' in rangeMap.keys():
        return list(range(rangeMap['_FROM'], rangeMap['_TO']+1, rangeMap['_STEP']))
    return list(range(rangeMap['_FROM'], rangeMap['_TO']+1))

def RF_HP_OPTIM_PROCESS (HP_OPTIM_INPUTS: dict, TRAIN_FILE_PATH: str, TEST_FILE_PATH: str, IN_PROGRESS: InProgressWindow):
    print(f'[RF_HYPERPARAMS_OPTIMIZE_PROCESS]:\n{jsonDumps(HP_OPTIM_INPUTS, indent=5)}')
    # PROCESS_PARAMS stores the FEATURES, METHOD, SCORING, CV
    PROCESS_PARAMS = {k:v for k,v in HP_OPTIM_INPUTS.items() if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM}
    trainDF = pd.read_excel(TRAIN_FILE_PATH)
    columnsToSelect = PROCESS_PARAMS['FEATURES']
    x_train = trainDF.loc[:, columnsToSelect]
    y_train = trainDF.iloc[:, -1]

    # GRID_SEARCH
    if PROCESS_PARAMS['METHOD']=='GridSearchCV':
        rfc_estimator = RandomForestClassifier(random_state=0)
        _PARAM_GRID = {}
        for k,v in HP_OPTIM_INPUTS.items():
            if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM:
                continue
            _PARAM_GRID[k] = _MAP_PARAMRANGE_TO_ACTUAL_LISTOFVALUES(v) if type(v)==dict else v
        print('PARAM_GRID: ', _PARAM_GRID)

        HP_OPTIM_METHOD_INSTANCE = GridSearchCV(
            estimator=rfc_estimator, param_grid=_PARAM_GRID, 
            scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], 
            verbose=2, n_jobs=4
        )
        HP_OPTIM_METHOD_INSTANCE.fit(x_train, y_train)

        xls_out = pd.DataFrame(HP_OPTIM_METHOD_INSTANCE.cv_results_)
        CHECK_DIR('output')
        xls_out.to_excel(
            excel_writer=os.path.join('output', f"RFC_GS-{PROCESS_PARAMS['SCORING']}.xlsx"), 
            index=False
        )
        results = HP_OPTIM_METHOD_INSTANCE.best_params_
        print('FINISHED RF_HP_OPTIM_PROCESS !!')

        # [CLEAN-UP after parallel processing]  
        # UserWarning: resource_tracker: There appear to be 10 leaked semlock objects to clean up at shutdown
        del HP_OPTIM_METHOD_INSTANCE
        hp_optim_gc.collect()
        return results

    elif PROCESS_PARAMS['METHOD']=='RandomizedSearchCV':
        rfc_estimator = RandomForestClassifier(random_state=0)
        _PARAM_GRID = {}
        for k,v in HP_OPTIM_INPUTS.items():
            if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM:
                continue
            _PARAM_GRID[k] = _MAP_PARAMRANGE_TO_ACTUAL_LISTOFVALUES(v) if type(v)==dict else v
        print('PARAM_GRID: ', _PARAM_GRID)

        HP_OPTIM_METHOD_INSTANCE = RandomizedSearchCV(
            estimator=rfc_estimator, param_distributions=_PARAM_GRID, 
            scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], 
            verbose=2, n_jobs=4
        )
        HP_OPTIM_METHOD_INSTANCE.fit(x_train, y_train)

        xls_out = pd.DataFrame(HP_OPTIM_METHOD_INSTANCE.cv_results_)
        CHECK_DIR('output')
        xls_out.to_excel(
            excel_writer=os.path.join('output', f"RFC_RS-{PROCESS_PARAMS['SCORING']}.xlsx"), 
            index=False
        )
        results = HP_OPTIM_METHOD_INSTANCE.best_params_
        print('FINISHED RF_HP_OPTIM_PROCESS !!')

        # [CLEAN-UP after parallel processing]  
        # UserWarning: resource_tracker: There appear to be 10 leaked semlock objects to clean up at shutdown
        del HP_OPTIM_METHOD_INSTANCE
        hp_optim_gc.collect()
        return results
        
    elif PROCESS_PARAMS['METHOD']=='Optuna':
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        _PARAM_GRID = {k:v for k,v in HP_OPTIM_INPUTS.items() if k not in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM}
        print('PARAM_GRID: ', _PARAM_GRID)

        def RF_OBJECTIVE (trial:optuna.Trial):
            n_estimators = trial.suggest_int(
                'n_estimators', 
                _PARAM_GRID['n_estimators']['_FROM'], _PARAM_GRID['n_estimators']['_TO'], step=_PARAM_GRID['n_estimators']['_STEP']
            )
            criterion = trial.suggest_categorical(
                'criterion', _PARAM_GRID['criterion']
            )
            max_depth = trial.suggest_int(
                'max_depth', 
                _PARAM_GRID['max_depth']['_FROM'], _PARAM_GRID['max_depth']['_TO']
            )
            min_samples_split = trial.suggest_int(
                'min_samples_split', 
                _PARAM_GRID['min_samples_split']['_FROM'], _PARAM_GRID['min_samples_split']['_TO']
            )
            min_samples_leaf = trial.suggest_int(
                'min_samples_leaf', 
                _PARAM_GRID['min_samples_leaf']['_FROM'], _PARAM_GRID['min_samples_leaf']['_TO']
            )
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                criterion=criterion,
                max_depth=max_depth, 
                min_samples_split=min_samples_split, 
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            # Perform cross-validation and return the average score
            return cross_val_score(
                estimator=model, X=x_train_scaled, y=y_train, 
                cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], scoring=PROCESS_PARAMS['SCORING']
            ).mean()
        
        IN_PROGRESS._CREATE_OPTUNA_PROGRESS()
        rf_study = optuna.create_study(direction='maximize', study_name='Random_Forest')
        rf_study.optimize(
            func=RF_OBJECTIVE, n_trials=OPTUNA_TOTAL_TRIALS, callbacks=[IN_PROGRESS._UPDATE_OPTUNA_PROGRESS_BAR]
        )
        IN_PROGRESS._COMPLETE_OPTUNA_PROGRESS()
        results = rf_study.best_params
        return results
    else:
        raise Exception(f"Unknown Method for HP_OPTIMIZATION: {PROCESS_PARAMS['METHOD']}")

def SVM_HP_OPTIM_PROCESS (HP_OPTIM_INPUTS: dict, TRAIN_FILE_PATH: str, TEST_FILE_PATH: str, IN_PROGRESS: InProgressWindow):
    print(f'[SVM_HYPERPARAMS_OPTIMIZE_PROCESS]:\n{jsonDumps(HP_OPTIM_INPUTS, indent=5)}')
    # PROCESS_PARAMS stores the FEATURES, METHOD, SCORING, CV
    PROCESS_PARAMS = {k:v for k,v in HP_OPTIM_INPUTS.items() if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM}
    trainDF = pd.read_excel(TRAIN_FILE_PATH)
    columnsToSelect = PROCESS_PARAMS['FEATURES']
    x_train = trainDF.loc[:, columnsToSelect]
    y_train = trainDF.iloc[:, -1]

    # GRID_SEARCH
    if PROCESS_PARAMS['METHOD']=='GridSearchCV':
        svc_estimator = SVC(random_state=0)
        _PARAM_GRID = {}
        for k,v in HP_OPTIM_INPUTS.items():
            if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM:
                continue
            _PARAM_GRID[k] = _MAP_PARAMRANGE_TO_ACTUAL_LISTOFVALUES(v) if type(v)==dict else v
        print('PARAM_GRID: ', _PARAM_GRID)

        HP_OPTIM_METHOD_INSTANCE = GridSearchCV(
            estimator=svc_estimator, param_grid=_PARAM_GRID, 
            scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], 
            verbose=2, n_jobs=4
        )
        HP_OPTIM_METHOD_INSTANCE.fit(x_train, y_train)

        xls_out = pd.DataFrame(HP_OPTIM_METHOD_INSTANCE.cv_results_)
        CHECK_DIR('output')
        xls_out.to_excel(
            excel_writer=os.path.join('output', f"SVM_GS-{PROCESS_PARAMS['SCORING']}.xlsx"), 
            index=False
        )
        results = HP_OPTIM_METHOD_INSTANCE.best_params_
        print('FINISHED RF_HP_OPTIM_PROCESS !!')

        # [CLEAN-UP after parallel processing]  
        # UserWarning: resource_tracker: There appear to be 10 leaked semlock objects to clean up at shutdown
        del HP_OPTIM_METHOD_INSTANCE
        hp_optim_gc.collect()
        return results

    elif PROCESS_PARAMS['METHOD']=='RandomizedSearchCV':
        svc_estimator = SVC(random_state=0)
        _PARAM_GRID = {}
        for k,v in HP_OPTIM_INPUTS.items():
            if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM:
                continue
            _PARAM_GRID[k] = _MAP_PARAMRANGE_TO_ACTUAL_LISTOFVALUES(v) if type(v)==dict else v
        print('PARAM_GRID: ', _PARAM_GRID)

        HP_OPTIM_METHOD_INSTANCE = RandomizedSearchCV(
            estimator=svc_estimator, param_distributions=_PARAM_GRID, 
            scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], 
            verbose=2, n_jobs=4
        )
        HP_OPTIM_METHOD_INSTANCE.fit(x_train, y_train)

        xls_out = pd.DataFrame(HP_OPTIM_METHOD_INSTANCE.cv_results_)
        CHECK_DIR('output')
        xls_out.to_excel(
            excel_writer=os.path.join('output', f"SVM_RS-{PROCESS_PARAMS['SCORING']}.xlsx"), 
            index=False
        )
        results = HP_OPTIM_METHOD_INSTANCE.best_params_
        print('FINISHED RF_HP_OPTIM_PROCESS !!')

        # [CLEAN-UP after parallel processing]  
        # UserWarning: resource_tracker: There appear to be 10 leaked semlock objects to clean up at shutdown
        del HP_OPTIM_METHOD_INSTANCE
        hp_optim_gc.collect()
        return results
        
    elif PROCESS_PARAMS['METHOD']=='Optuna':
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        _PARAM_GRID = {k:v for k,v in HP_OPTIM_INPUTS.items() if k not in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM}
        print('PARAM_GRID: ', _PARAM_GRID)

        def SVM_OBJECTIVE (trial:optuna.Trial):
            C = trial.suggest_categorical(
                'C', _PARAM_GRID['C']
            )
            kernel = trial.suggest_categorical(
                'kernel', _PARAM_GRID['kernel']
            )
            gamma = trial.suggest_categorical(
                'gamma', _PARAM_GRID['gamma']
            )
            model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
            # Perform cross-validation and return the average score
            return cross_val_score(
                estimator=model, X=x_train_scaled, y=y_train, 
                cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], scoring=PROCESS_PARAMS['SCORING']
            ).mean()
        
        IN_PROGRESS._CREATE_OPTUNA_PROGRESS()
        svm_study = optuna.create_study(direction='maximize', study_name='SupportVectorMachine')
        svm_study.optimize(
            func=SVM_OBJECTIVE, n_trials=OPTUNA_TOTAL_TRIALS, callbacks=[IN_PROGRESS._UPDATE_OPTUNA_PROGRESS_BAR]
        )
        IN_PROGRESS._COMPLETE_OPTUNA_PROGRESS()
        results = svm_study.best_params
        return results
    else:
        raise Exception(f"Unknown Method for HP_OPTIMIZATION: {PROCESS_PARAMS['METHOD']}")

def LDA_HP_OPTIM_PROCESS (HP_OPTIM_INPUTS: dict, TRAIN_FILE_PATH: str, TEST_FILE_PATH: str, IN_PROGRESS: InProgressWindow):
    print(f'[LDA_HYPERPARAMS_OPTIMIZE_PROCESS]:\n{jsonDumps(HP_OPTIM_INPUTS, indent=5)}')
    # PROCESS_PARAMS stores the FEATURES, METHOD, SCORING, CV
    PROCESS_PARAMS = {k:v for k,v in HP_OPTIM_INPUTS.items() if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM}
    trainDF = pd.read_excel(TRAIN_FILE_PATH)
    columnsToSelect = PROCESS_PARAMS['FEATURES']
    x_train = trainDF.loc[:, columnsToSelect]
    y_train = trainDF.iloc[:, -1]

    # GRID_SEARCH
    if PROCESS_PARAMS['METHOD']=='GridSearchCV':
        lda_estimator = LinearDiscriminantAnalysis()
        _PARAM_GRID = {}
        for k,v in HP_OPTIM_INPUTS.items():
            if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM:
                continue
            _PARAM_GRID[k] = _MAP_PARAMRANGE_TO_ACTUAL_LISTOFVALUES(v) if type(v)==dict else v
        print('PARAM_GRID: ', _PARAM_GRID)

        HP_OPTIM_METHOD_INSTANCE = GridSearchCV(
            estimator=lda_estimator, param_grid=_PARAM_GRID, 
            scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], 
            verbose=2, n_jobs=4
        )
        HP_OPTIM_METHOD_INSTANCE.fit(x_train, y_train)

        xls_out = pd.DataFrame(HP_OPTIM_METHOD_INSTANCE.cv_results_)
        CHECK_DIR('output')
        xls_out.to_excel(
            excel_writer=os.path.join('output', f"LDA_GS-{PROCESS_PARAMS['SCORING']}.xlsx"), 
            index=False
        )
        results = HP_OPTIM_METHOD_INSTANCE.best_params_
        print('FINISHED LDA_HP_OPTIM_PROCESS !!')

        # [CLEAN-UP after parallel processing]  
        # UserWarning: resource_tracker: There appear to be 10 leaked semlock objects to clean up at shutdown
        del HP_OPTIM_METHOD_INSTANCE
        hp_optim_gc.collect()
        return results

    elif PROCESS_PARAMS['METHOD']=='RandomizedSearchCV':
        lda_estimator = LinearDiscriminantAnalysis()
        _PARAM_GRID = {}
        for k,v in HP_OPTIM_INPUTS.items():
            if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM:
                continue
            _PARAM_GRID[k] = _MAP_PARAMRANGE_TO_ACTUAL_LISTOFVALUES(v) if type(v)==dict else v
        print('PARAM_GRID: ', _PARAM_GRID)

        HP_OPTIM_METHOD_INSTANCE = RandomizedSearchCV(
            estimator=lda_estimator, param_distributions=_PARAM_GRID, 
            scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], 
            verbose=2, n_jobs=4
        )
        HP_OPTIM_METHOD_INSTANCE.fit(x_train, y_train)

        xls_out = pd.DataFrame(HP_OPTIM_METHOD_INSTANCE.cv_results_)
        CHECK_DIR('output')
        xls_out.to_excel(
            excel_writer=os.path.join('output', f"LDA_RS-{PROCESS_PARAMS['SCORING']}.xlsx"), 
            index=False
        )
        results = HP_OPTIM_METHOD_INSTANCE.best_params_
        print('FINISHED LDA_HP_OPTIM_PROCESS !!')

        # [CLEAN-UP after parallel processing]  
        # UserWarning: resource_tracker: There appear to be 10 leaked semlock objects to clean up at shutdown
        del HP_OPTIM_METHOD_INSTANCE
        hp_optim_gc.collect()
        return results
        
    elif PROCESS_PARAMS['METHOD']=='Optuna':
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        _PARAM_GRID = {k:v for k,v in HP_OPTIM_INPUTS.items() if k not in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM}
        print('PARAM_GRID: ', _PARAM_GRID)

        def LDA_OBJECTIVE (trial):
            solver = trial.suggest_categorical("solver", _PARAM_GRID['solver'])
            shrinkage = None
            if solver in ['lsqr', 'eigen']:
                shrinkage = trial.suggest_float('shrinkage', 0.0, 1.0) if solver == 'lsqr' else None
            model = LinearDiscriminantAnalysis(
                solver=solver, 
                shrinkage=shrinkage
            )
            return cross_val_score(
                model, x_train_scaled, y_train, 
                scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID']
            ).mean()
        
        IN_PROGRESS._CREATE_OPTUNA_PROGRESS()
        lda_study = optuna.create_study(direction='maximize', study_name='LinearDiscriminantAnalysis')
        lda_study.optimize(
            func=LDA_OBJECTIVE, n_trials=OPTUNA_TOTAL_TRIALS*10, callbacks=[IN_PROGRESS._UPDATE_OPTUNA_PROGRESS_BAR]
        )
        IN_PROGRESS._COMPLETE_OPTUNA_PROGRESS()
        results = lda_study.best_params
        return results
    else:
        raise Exception(f"Unknown Method for HP_OPTIMIZATION: {PROCESS_PARAMS['METHOD']}")

def LR_HP_OPTIM_PROCESS (HP_OPTIM_INPUTS: dict, TRAIN_FILE_PATH: str, TEST_FILE_PATH: str, IN_PROGRESS: InProgressWindow):
    print(f'[LR_HYPERPARAMS_OPTIMIZE_PROCESS]:\n{jsonDumps(HP_OPTIM_INPUTS, indent=5)}')
    # PROCESS_PARAMS stores the FEATURES, METHOD, SCORING, CV
    PROCESS_PARAMS = {k:v for k,v in HP_OPTIM_INPUTS.items() if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM}
    trainDF = pd.read_excel(TRAIN_FILE_PATH)
    columnsToSelect = PROCESS_PARAMS['FEATURES']
    x_train = trainDF.loc[:, columnsToSelect]
    y_train = trainDF.iloc[:, -1]

    # GRID_SEARCH
    if PROCESS_PARAMS['METHOD']=='GridSearchCV':
        lr_estimator = LogisticRegression(random_state=0)
        _PARAM_GRID = {}
        for k,v in HP_OPTIM_INPUTS.items():
            if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM:
                continue
            _PARAM_GRID[k] = _MAP_PARAMRANGE_TO_ACTUAL_LISTOFVALUES(v) if type(v)==dict else v
        print('PARAM_GRID: ', _PARAM_GRID)

        HP_OPTIM_METHOD_INSTANCE = GridSearchCV(
            estimator=lr_estimator, param_grid=_PARAM_GRID, 
            scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], 
            verbose=2, n_jobs=4
        )
        HP_OPTIM_METHOD_INSTANCE.fit(x_train, y_train)

        xls_out = pd.DataFrame(HP_OPTIM_METHOD_INSTANCE.cv_results_)
        CHECK_DIR('output')
        xls_out.to_excel(
            excel_writer=os.path.join('output', f"LR_GS-{PROCESS_PARAMS['SCORING']}.xlsx"), 
            index=False
        )
        results = HP_OPTIM_METHOD_INSTANCE.best_params_
        print('FINISHED LR_HP_OPTIM_PROCESS !!')

        # [CLEAN-UP after parallel processing]  
        # UserWarning: resource_tracker: There appear to be 10 leaked semlock objects to clean up at shutdown
        del HP_OPTIM_METHOD_INSTANCE
        hp_optim_gc.collect()
        return results

    elif PROCESS_PARAMS['METHOD']=='RandomizedSearchCV':
        lr_estimator = LogisticRegression(random_state=0)
        _PARAM_GRID = {}
        for k,v in HP_OPTIM_INPUTS.items():
            if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM:
                continue
            _PARAM_GRID[k] = _MAP_PARAMRANGE_TO_ACTUAL_LISTOFVALUES(v) if type(v)==dict else v
        print('PARAM_GRID: ', _PARAM_GRID)

        HP_OPTIM_METHOD_INSTANCE = RandomizedSearchCV(
            estimator=lr_estimator, param_distributions=_PARAM_GRID, 
            scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], 
            verbose=2, n_jobs=4
        )
        HP_OPTIM_METHOD_INSTANCE.fit(x_train, y_train)

        xls_out = pd.DataFrame(HP_OPTIM_METHOD_INSTANCE.cv_results_)
        CHECK_DIR('output')
        xls_out.to_excel(
            excel_writer=os.path.join('output', f"LR_RS-{PROCESS_PARAMS['SCORING']}.xlsx"), 
            index=False
        )
        results = HP_OPTIM_METHOD_INSTANCE.best_params_
        print('FINISHED LR_HP_OPTIM_PROCESS !!')

        # [CLEAN-UP after parallel processing]  
        # UserWarning: resource_tracker: There appear to be 10 leaked semlock objects to clean up at shutdown
        del HP_OPTIM_METHOD_INSTANCE
        hp_optim_gc.collect()
        return results
        
    elif PROCESS_PARAMS['METHOD']=='Optuna':
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        _PARAM_GRID = {k:v for k,v in HP_OPTIM_INPUTS.items() if k not in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM}
        print('PARAM_GRID: ', _PARAM_GRID)

        def LR_OBJECTIVE (trial):
            C = trial.suggest_categorical(
                'C', _PARAM_GRID['C']
            )
            solver = trial.suggest_categorical("solver", _PARAM_GRID['solver'])
            penalty = trial.suggest_categorical("penalty", _PARAM_GRID['penalty'])

            # Valid solver-penalty combinations
            valid_combinations = {
                "liblinear": ["l1", "l2"],
                "saga": ["l1", "l2", "elasticnet", "none"],
                "lbfgs": ["l2", "none"],
                "newton-cg": ["l2", "none"],
                "newton-cholesky": ["l2", "none"],
                "sag": ["l2", "none"],
            }
            
            if penalty not in valid_combinations[solver]:
                raise optuna.TrialPruned()

            l1_ratio = None
            if penalty == "elasticnet":
                l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
                model = LogisticRegression(
                    C=C, penalty=penalty, solver=solver, l1_ratio=l1_ratio, max_iter=10000, random_state=42
                )
            else:
                model = LogisticRegression(
                    C=C, penalty=penalty, solver=solver, max_iter=10000, random_state=42
                )
            return cross_val_score(
                model, x_train_scaled, y_train, 
                scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID']
            ).mean()
        
        IN_PROGRESS._CREATE_OPTUNA_PROGRESS()
        lr_study = optuna.create_study(direction='maximize', study_name='LogisticRegression')
        lr_study.optimize(
            func=LR_OBJECTIVE, n_trials=OPTUNA_TOTAL_TRIALS*10, callbacks=[IN_PROGRESS._UPDATE_OPTUNA_PROGRESS_BAR]
        )
        IN_PROGRESS._COMPLETE_OPTUNA_PROGRESS()
        results = lr_study.best_params
        return results
    else:
        raise Exception(f"Unknown Method for HP_OPTIMIZATION: {PROCESS_PARAMS['METHOD']}")

def KNN_HP_OPTIM_PROCESS (HP_OPTIM_INPUTS: dict, TRAIN_FILE_PATH: str, TEST_FILE_PATH: str, IN_PROGRESS: InProgressWindow):
    print(f'[KNN_HYPERPARAMS_OPTIMIZE_PROCESS]:\n{jsonDumps(HP_OPTIM_INPUTS, indent=5)}')
    # PROCESS_PARAMS stores the FEATURES, METHOD, SCORING, CV
    PROCESS_PARAMS = {k:v for k,v in HP_OPTIM_INPUTS.items() if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM}
    trainDF = pd.read_excel(TRAIN_FILE_PATH)
    columnsToSelect = PROCESS_PARAMS['FEATURES']
    x_train = trainDF.loc[:, columnsToSelect]
    y_train = trainDF.iloc[:, -1]

    # GRID_SEARCH
    if PROCESS_PARAMS['METHOD']=='GridSearchCV':
        knn_estimator = KNeighborsClassifier()
        _PARAM_GRID = {}
        for k,v in HP_OPTIM_INPUTS.items():
            if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM:
                continue
            _PARAM_GRID[k] = _MAP_PARAMRANGE_TO_ACTUAL_LISTOFVALUES(v) if type(v)==dict else v
        print('PARAM_GRID: ', _PARAM_GRID)

        HP_OPTIM_METHOD_INSTANCE = GridSearchCV(
            estimator=knn_estimator, param_grid=_PARAM_GRID, 
            scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], 
            verbose=2, n_jobs=4
        )
        HP_OPTIM_METHOD_INSTANCE.fit(x_train, y_train)

        xls_out = pd.DataFrame(HP_OPTIM_METHOD_INSTANCE.cv_results_)
        CHECK_DIR('output')
        xls_out.to_excel(
            excel_writer=os.path.join('output', f"KNN_GS-{PROCESS_PARAMS['SCORING']}.xlsx"), 
            index=False
        )
        results = HP_OPTIM_METHOD_INSTANCE.best_params_
        print('FINISHED KNN_HP_OPTIM_PROCESS !!')

        # [CLEAN-UP after parallel processing]  
        # UserWarning: resource_tracker: There appear to be 10 leaked semlock objects to clean up at shutdown
        del HP_OPTIM_METHOD_INSTANCE
        hp_optim_gc.collect()
        return results

    elif PROCESS_PARAMS['METHOD']=='RandomizedSearchCV':
        knn_estimator = KNeighborsClassifier()
        _PARAM_GRID = {}
        for k,v in HP_OPTIM_INPUTS.items():
            if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM:
                continue
            _PARAM_GRID[k] = _MAP_PARAMRANGE_TO_ACTUAL_LISTOFVALUES(v) if type(v)==dict else v
        print('PARAM_GRID: ', _PARAM_GRID)

        HP_OPTIM_METHOD_INSTANCE = RandomizedSearchCV(
            estimator=knn_estimator, param_distributions=_PARAM_GRID, 
            scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], 
            verbose=2, n_jobs=4
        )
        HP_OPTIM_METHOD_INSTANCE.fit(x_train, y_train)

        xls_out = pd.DataFrame(HP_OPTIM_METHOD_INSTANCE.cv_results_)
        CHECK_DIR('output')
        xls_out.to_excel(
            excel_writer=os.path.join('output', f"KNN_RS-{PROCESS_PARAMS['SCORING']}.xlsx"), 
            index=False
        )
        results = HP_OPTIM_METHOD_INSTANCE.best_params_
        print('FINISHED KNN_HP_OPTIM_PROCESS !!')

        # [CLEAN-UP after parallel processing]  
        # UserWarning: resource_tracker: There appear to be 10 leaked semlock objects to clean up at shutdown
        del HP_OPTIM_METHOD_INSTANCE
        hp_optim_gc.collect()
        return results
        
    elif PROCESS_PARAMS['METHOD']=='Optuna':
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        _PARAM_GRID = {k:v for k,v in HP_OPTIM_INPUTS.items() if k not in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM}
        print('PARAM_GRID: ', _PARAM_GRID)

        def KNN_OBJECTIVE (trial):
            n_neighbors = trial.suggest_int(
                'n_neighbors', _PARAM_GRID['n_neighbors']['_FROM'], _PARAM_GRID['n_neighbors']['_TO']
            )
            leaf_size = trial.suggest_int(
                'leaf_size', _PARAM_GRID['leaf_size']['_FROM'], _PARAM_GRID['leaf_size']['_TO']
            )
            p = trial.suggest_int(
                'p', _PARAM_GRID['p']['_FROM'], _PARAM_GRID['p']['_TO']
            )
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors, 
                leaf_size=leaf_size, 
                p=p
            )
            return cross_val_score(
                model, x_train_scaled, y_train, 
                scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID']
            ).mean()
        
        IN_PROGRESS._CREATE_OPTUNA_PROGRESS()
        knn_study = optuna.create_study(direction='maximize', study_name='KNearestNeighbors')
        knn_study.optimize(
            func=KNN_OBJECTIVE, n_trials=OPTUNA_TOTAL_TRIALS*10, callbacks=[IN_PROGRESS._UPDATE_OPTUNA_PROGRESS_BAR]
        )
        IN_PROGRESS._COMPLETE_OPTUNA_PROGRESS()
        results = knn_study.best_params
        return results
    else:
        raise Exception(f"Unknown Method for HP_OPTIMIZATION: {PROCESS_PARAMS['METHOD']}")

def GB_HP_OPTIM_PROCESS (HP_OPTIM_INPUTS: dict, TRAIN_FILE_PATH: str, TEST_FILE_PATH: str, IN_PROGRESS: InProgressWindow):
    print(f'[GB_HYPERPARAMS_OPTIMIZE_PROCESS]:\n{jsonDumps(HP_OPTIM_INPUTS, indent=5)}')
    # PROCESS_PARAMS stores the FEATURES, METHOD, SCORING, CV
    PROCESS_PARAMS = {k:v for k,v in HP_OPTIM_INPUTS.items() if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM}
    trainDF = pd.read_excel(TRAIN_FILE_PATH)
    columnsToSelect = PROCESS_PARAMS['FEATURES']
    x_train = trainDF.loc[:, columnsToSelect]
    y_train = trainDF.iloc[:, -1]

    # GRID_SEARCH
    if PROCESS_PARAMS['METHOD']=='GridSearchCV':
        gb_estimator = GradientBoostingClassifier(random_state=0)
        _PARAM_GRID = {}
        for k,v in HP_OPTIM_INPUTS.items():
            if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM:
                continue
            _PARAM_GRID[k] = _MAP_PARAMRANGE_TO_ACTUAL_LISTOFVALUES(v) if type(v)==dict else v
        print('PARAM_GRID: ', _PARAM_GRID)

        HP_OPTIM_METHOD_INSTANCE = GridSearchCV(
            estimator=gb_estimator, param_grid=_PARAM_GRID, 
            scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], 
            verbose=2, n_jobs=4
        )
        HP_OPTIM_METHOD_INSTANCE.fit(x_train, y_train)

        xls_out = pd.DataFrame(HP_OPTIM_METHOD_INSTANCE.cv_results_)
        CHECK_DIR('output')
        xls_out.to_excel(
            excel_writer=os.path.join('output', f"RFC_GS-{PROCESS_PARAMS['SCORING']}.xlsx"), 
            index=False
        )
        results = HP_OPTIM_METHOD_INSTANCE.best_params_
        print('FINISHED GB_HP_OPTIM_PROCESS !!')

        # [CLEAN-UP after parallel processing]  
        # UserWarning: resource_tracker: There appear to be 10 leaked semlock objects to clean up at shutdown
        del HP_OPTIM_METHOD_INSTANCE
        hp_optim_gc.collect()
        return results

    elif PROCESS_PARAMS['METHOD']=='RandomizedSearchCV':
        gb_estimator = GradientBoostingClassifier(random_state=0)
        _PARAM_GRID = {}
        for k,v in HP_OPTIM_INPUTS.items():
            if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM:
                continue
            _PARAM_GRID[k] = _MAP_PARAMRANGE_TO_ACTUAL_LISTOFVALUES(v) if type(v)==dict else v
        print('PARAM_GRID: ', _PARAM_GRID)

        HP_OPTIM_METHOD_INSTANCE = RandomizedSearchCV(
            estimator=gb_estimator, param_distributions=_PARAM_GRID, 
            scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], 
            verbose=2, n_jobs=4
        )
        HP_OPTIM_METHOD_INSTANCE.fit(x_train, y_train)

        xls_out = pd.DataFrame(HP_OPTIM_METHOD_INSTANCE.cv_results_)
        CHECK_DIR('output')
        xls_out.to_excel(
            excel_writer=os.path.join('output', f"RFC_RS-{PROCESS_PARAMS['SCORING']}.xlsx"), 
            index=False
        )
        results = HP_OPTIM_METHOD_INSTANCE.best_params_
        print('FINISHED GB_HP_OPTIM_PROCESS !!')

        # [CLEAN-UP after parallel processing]  
        # UserWarning: resource_tracker: There appear to be 10 leaked semlock objects to clean up at shutdown
        del HP_OPTIM_METHOD_INSTANCE
        hp_optim_gc.collect()
        return results
        
    elif PROCESS_PARAMS['METHOD']=='Optuna':
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        _PARAM_GRID = {k:v for k,v in HP_OPTIM_INPUTS.items() if k not in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM}
        print('PARAM_GRID: ', _PARAM_GRID)

        def GB_OBJECTIVE (trial:optuna.Trial):
            n_estimators = trial.suggest_int(
                'n_estimators', 
                _PARAM_GRID['n_estimators']['_FROM'], _PARAM_GRID['n_estimators']['_TO'], step=_PARAM_GRID['n_estimators']['_STEP']
            )
            learning_rate = trial.suggest_float(
                'learning_rate', 
                _PARAM_GRID['learning_rate']['_FROM'], _PARAM_GRID['learning_rate']['_TO'],
            )
            criterion = trial.suggest_categorical(
                'criterion', _PARAM_GRID['criterion']
            )
            max_depth = trial.suggest_int(
                'max_depth', 
                _PARAM_GRID['max_depth']['_FROM'], _PARAM_GRID['max_depth']['_TO']
            )
            min_samples_split = trial.suggest_int(
                'min_samples_split', 
                _PARAM_GRID['min_samples_split']['_FROM'], _PARAM_GRID['min_samples_split']['_TO']
            )
            min_samples_leaf = trial.suggest_int(
                'min_samples_leaf', 
                _PARAM_GRID['min_samples_leaf']['_FROM'], _PARAM_GRID['min_samples_leaf']['_TO']
            )
            model = GradientBoostingClassifier(
                n_estimators=n_estimators, 
                learning_rate=learning_rate,
                criterion=criterion,
                max_depth=max_depth, 
                min_samples_split=min_samples_split, 
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            # Perform cross-validation and return the average score
            return cross_val_score(
                estimator=model, X=x_train_scaled, y=y_train, 
                cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], scoring=PROCESS_PARAMS['SCORING']
            ).mean()
        
        IN_PROGRESS._CREATE_OPTUNA_PROGRESS()
        gb_study = optuna.create_study(direction='maximize', study_name='GradientBoosting')
        gb_study.optimize(
            func=GB_OBJECTIVE, n_trials=OPTUNA_TOTAL_TRIALS, callbacks=[IN_PROGRESS._UPDATE_OPTUNA_PROGRESS_BAR]
        )
        IN_PROGRESS._COMPLETE_OPTUNA_PROGRESS()
        results = gb_study.best_params
        return results
    else:
        raise Exception(f"Unknown Method for HP_OPTIMIZATION: {PROCESS_PARAMS['METHOD']}")

def MLP_HP_OPTIM_PROCESS (HP_OPTIM_INPUTS: dict, TRAIN_FILE_PATH: str, TEST_FILE_PATH: str, IN_PROGRESS: InProgressWindow):
    print(f'[MLP_HYPERPARAMS_OPTIMIZE_PROCESS]:\n{jsonDumps(HP_OPTIM_INPUTS, indent=5)}')

    # PROCESS_PARAMS stores the FEATURES, METHOD, SCORING, CV
    PROCESS_PARAMS = {k:v for k,v in HP_OPTIM_INPUTS.items() if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM}
    trainDF = pd.read_excel(TRAIN_FILE_PATH)
    columnsToSelect = PROCESS_PARAMS['FEATURES']
    x_train = trainDF.loc[:, columnsToSelect]
    y_train = trainDF.iloc[:, -1]
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    # GRID_SEARCH
    if PROCESS_PARAMS['METHOD']=='GridSearchCV':
        mlp_estimator = MLPClassifier(max_iter=1000, random_state=None, verbose=True)
        _PARAM_GRID = {}
        for k,v in HP_OPTIM_INPUTS.items():
            if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM:
                continue
            _PARAM_GRID[k] = _MAP_PARAMRANGE_TO_ACTUAL_LISTOFVALUES(v) if type(v)==dict else v

        print('sss:',_PARAM_GRID)
        
        # FIND all HIDDEN_LAYER_COMBINATIONS
        values = _PARAM_GRID['hidden_layer_size']
        n_values = _PARAM_GRID['num_of_hidden_layers']

        del[_PARAM_GRID['hidden_layer_size']]
        del[_PARAM_GRID['num_of_hidden_layers']]

        _PARAM_GRID['hidden_layer_sizes'] = [tuple(combo) for n in n_values for combo in product(values, repeat=n)]
        _PARAM_GRID['alpha'] = list(map(float, _PARAM_GRID['alpha']))

        print('PARAM_GRID: ', _PARAM_GRID)

        HP_OPTIM_METHOD_INSTANCE = GridSearchCV(
            estimator=mlp_estimator, param_grid=_PARAM_GRID, 
            scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], 
            verbose=3, n_jobs=-1
        )
        HP_OPTIM_METHOD_INSTANCE.fit(x_train_scaled, y_train) # NOT SCALED !!

        xls_out = pd.DataFrame(HP_OPTIM_METHOD_INSTANCE.cv_results_)
        CHECK_DIR('output')
        xls_out.to_excel(
            excel_writer=os.path.join('output', f"MLP_GS-{PROCESS_PARAMS['SCORING']}.xlsx"), 
            index=False
        )
        results = HP_OPTIM_METHOD_INSTANCE.best_params_
        print('FINISHED MLP_HP_OPTIM_PROCESS !!')

        # [CLEAN-UP after parallel processing]  
        # UserWarning: resource_tracker: There appear to be 10 leaked semlock objects to clean up at shutdown
        del HP_OPTIM_METHOD_INSTANCE
        hp_optim_gc.collect()
        return results

    elif PROCESS_PARAMS['METHOD']=='RandomizedSearchCV':
        mlp_estimator = MLPClassifier(max_iter=1000, random_state=None, verbose=True)
        _PARAM_GRID = {}
        for k,v in HP_OPTIM_INPUTS.items():
            if k in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM:
                continue
            _PARAM_GRID[k] = _MAP_PARAMRANGE_TO_ACTUAL_LISTOFVALUES(v) if type(v)==dict else v
        
        # FIND all HIDDEN_LAYER_COMBINATIONS
        values = _PARAM_GRID['hidden_layer_size']
        n_values = _PARAM_GRID['num_of_hidden_layers']

        del[_PARAM_GRID['hidden_layer_size']]
        del[_PARAM_GRID['num_of_hidden_layers']]

        _PARAM_GRID['hidden_layer_sizes'] = [tuple(combo) for n in n_values for combo in product(values, repeat=n)]
        _PARAM_GRID['alpha'] = list(map(int, _PARAM_GRID['alpha']))

        print('PARAM_GRID: ', _PARAM_GRID)

        HP_OPTIM_METHOD_INSTANCE = RandomizedSearchCV(
            estimator=mlp_estimator, param_distributions=_PARAM_GRID, 
            scoring=PROCESS_PARAMS['SCORING'], cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], 
            verbose=3, n_jobs=-1
        )
        HP_OPTIM_METHOD_INSTANCE.fit(x_train_scaled, y_train)

        xls_out = pd.DataFrame(HP_OPTIM_METHOD_INSTANCE.cv_results_)
        CHECK_DIR('output')
        xls_out.to_excel(
            excel_writer=os.path.join('output', f"MLP_RS-{PROCESS_PARAMS['SCORING']}.xlsx"), 
            index=False
        )
        results = HP_OPTIM_METHOD_INSTANCE.best_params_
        print('FINISHED MLP_HP_OPTIM_PROCESS !!')

        # [CLEAN-UP after parallel processing]  
        # UserWarning: resource_tracker: There appear to be 10 leaked semlock objects to clean up at shutdown
        del HP_OPTIM_METHOD_INSTANCE
        hp_optim_gc.collect()
        return results
        
    elif PROCESS_PARAMS['METHOD']=='Optuna':
        _PARAM_GRID = {k:v for k,v in HP_OPTIM_INPUTS.items() if k not in PARAM_GRID_KEYS_NOT_FOR_HP_OPTIM}
        print('PARAM_GRID: ', _PARAM_GRID)

        def MLP_OBJECTIVE (trial:optuna.Trial):
            hidden_layer_sizes = tuple([
                trial.suggest_categorical(
                    f"n_layer_{i}", 
                    list(range(_PARAM_GRID['hidden_layer_size']['_FROM'], _PARAM_GRID['hidden_layer_size']['_TO'], _PARAM_GRID['hidden_layer_size']['_STEP']))
                ) 
                for i in range(trial.suggest_int(
                    "n_layers", 
                    _PARAM_GRID['num_of_hidden_layers']['_FROM'],  _PARAM_GRID['num_of_hidden_layers']['_TO']
                ))
            ])
            activation = trial.suggest_categorical(
                "activation", _PARAM_GRID['activation']
            )
            solver = trial.suggest_categorical(
                "solver", _PARAM_GRID['solver']
            )
            alpha = trial.suggest_categorical(
                "alpha", list(map(float, _PARAM_GRID['alpha']))
            )
            learning_rate = trial.suggest_categorical(
                "learning_rate", _PARAM_GRID['learning_rate']
            )

            # Create the MLPClassifier with the sampled hyperparameters
            model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                alpha=alpha,
                learning_rate=learning_rate,
                max_iter=1000,  # You can adjust the number of iterations
                random_state=42,
                verbose=False
            )

            # Perform cross-validation and return the average score
            return cross_val_score(
                estimator=model, X=x_train_scaled, y=y_train, 
                cv=PROCESS_PARAMS['CROSS_FOLD_VALID'], scoring=PROCESS_PARAMS['SCORING']
            ).mean()
        
        IN_PROGRESS._CREATE_OPTUNA_PROGRESS()
        mlp_study = optuna.create_study(direction='maximize', study_name='MLP')
        mlp_study.optimize(
            func=MLP_OBJECTIVE, n_trials=OPTUNA_TOTAL_TRIALS, callbacks=[IN_PROGRESS._UPDATE_OPTUNA_PROGRESS_BAR]
        )
        IN_PROGRESS._COMPLETE_OPTUNA_PROGRESS()
        results = mlp_study.best_params
        print(mlp_study.best_params)

        # Re-build the hidden_layer_sizes param
        hidden_layer_sizes = []
        for layer_idx in range(results['n_layers']):
            hidden_layer_sizes.append(results[f'n_layer_{layer_idx}'])
            del[results[f'n_layer_{layer_idx}']]
        del[results['n_layers']]
        results['hidden_layer_sizes'] = tuple(hidden_layer_sizes)
        return results
    else:
        raise Exception(f"Unknown Method for HP_OPTIMIZATION: {PROCESS_PARAMS['METHOD']}")

#---------MODEL BUILD -----------

def RF_MODEL_BUILD_PROCESS (RfMB_ValidatedInputs: dict, trainFilePath: str, testFilePath: str):
    FEATURES_LIST = RfMB_ValidatedInputs["FEATURES"]
    del(RfMB_ValidatedInputs["FEATURES"])
    rfc_regressor = RandomForestClassifier(**RfMB_ValidatedInputs, oob_score=True)
    
    results = MODEL_BUILD_GENERATE_RESULTS(
        regressor=rfc_regressor, 
        featureList=FEATURES_LIST,
        trainFilePath=trainFilePath,
        testFilePath=testFilePath,
        regressorName='RFC'
    )
    return results

def GB_MODEL_BUILD_PROCESS (GbMB_ValidatedInputs: dict, trainFilePath: str, testFilePath: str):
    FEATURES_LIST = GbMB_ValidatedInputs["FEATURES"]
    del(GbMB_ValidatedInputs["FEATURES"])
    gb_regressor = GradientBoostingClassifier(**GbMB_ValidatedInputs)
    
    results = MODEL_BUILD_GENERATE_RESULTS(
        regressor=gb_regressor, 
        featureList=FEATURES_LIST,
        trainFilePath=trainFilePath,
        testFilePath=testFilePath,
        regressorName='GB'
    )
    return results

def SVM_MODEL_BUILD_PROCESS (SvmMB_ValidatedInputs: dict, trainFilePath: str, testFilePath: str):
    FEATURES_LIST = SvmMB_ValidatedInputs["FEATURES"]
    del[SvmMB_ValidatedInputs['FEATURES']]
    svm_regressor = SVC(**SvmMB_ValidatedInputs)

    results = MODEL_BUILD_GENERATE_RESULTS(
        regressor=svm_regressor, 
        featureList=FEATURES_LIST,
        trainFilePath=trainFilePath,
        testFilePath=testFilePath,
        regressorName='SVC'
    )
    return results

def LR_MODEL_BUILD_PROCESS (LrMB_ValidatedInputs: dict, trainFilePath: str, testFilePath: str):
    FEATURES_LIST = LrMB_ValidatedInputs["FEATURES"]
    del[LrMB_ValidatedInputs['FEATURES']]
    lr_regressor = LogisticRegression(**LrMB_ValidatedInputs)

    results = MODEL_BUILD_GENERATE_RESULTS(
        regressor=lr_regressor, 
        featureList=FEATURES_LIST,
        trainFilePath=trainFilePath,
        testFilePath=testFilePath,
        regressorName='LR'
    )
    return results

def LDA_MODEL_BUILD_PROCESS (LdaMB_ValidatedInputs: dict, trainFilePath: str, testFilePath: str):
    FEATURES_LIST = LdaMB_ValidatedInputs["FEATURES"]
    del[LdaMB_ValidatedInputs['FEATURES']]
    lda_regressor = LinearDiscriminantAnalysis(**LdaMB_ValidatedInputs)

    results = MODEL_BUILD_GENERATE_RESULTS(
        regressor=lda_regressor, 
        featureList=FEATURES_LIST,
        trainFilePath=trainFilePath,
        testFilePath=testFilePath,
        regressorName='LDA'
    )
    return results

def KNN_MODEL_BUILD_PROCESS (KnnMB_ValidatedInputs: dict, trainFilePath: str, testFilePath: str):
    FEATURES_LIST = KnnMB_ValidatedInputs["FEATURES"]
    del[KnnMB_ValidatedInputs['FEATURES']]
    knn_regressor = KNeighborsClassifier(**KnnMB_ValidatedInputs)

    results = MODEL_BUILD_GENERATE_RESULTS(
        regressor=knn_regressor, 
        featureList=FEATURES_LIST,
        trainFilePath=trainFilePath,
        testFilePath=testFilePath,
        regressorName='KNN'
    )
    return results

def MLP_MODEL_BUILD_PROCESS (MlpMB_ValidatedInputs: dict, trainFilePath: str, testFilePath: str):
    #print(f'[MLP_MODEL_BUILD_PROCESS]:\n{jsonDumps(MlpMB_ValidatedInputs, indent=5)}')
    FEATURES_LIST = MlpMB_ValidatedInputs["FEATURES"]
    del[MlpMB_ValidatedInputs['FEATURES']]
    mlp_regressor = MLPClassifier(**MlpMB_ValidatedInputs)

    results = MODEL_BUILD_GENERATE_RESULTS(
        regressor=mlp_regressor, 
        featureList=FEATURES_LIST,
        trainFilePath=trainFilePath,
        testFilePath=testFilePath,
        regressorName='MLP'
    )
    return results