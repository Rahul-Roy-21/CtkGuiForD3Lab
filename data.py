_COMMON_PROPS = {
    'feature_selection': {
        'ranking_method': 'MIS', # MDF or MIS allowed
        'min_features_selected': 12,
    },
    'hp_optim': {
        'method_opts_without_Optuna': ['GridSearchCV','RandomizedSearchCV'],
        'method_opts_with_Optuna': ['Optuna', 'GridSearchCV','RandomizedSearchCV'],
        'scoring_opts': [
            'accuracy', 'balanced_accuracy','average_precision','f1','f1_micro', 'f1_macro', 'f1_weighted','precision', 'recall', 'jaccard','roc_auc', 'roc_auc_ovr','roc_auc_ovo', 'roc_auc_ovr_weighted','roc_auc_ovo_weighted'
        ],
        'optuna_total_trials' : 40,
    }
}

DATA = {
    "app_name": "D3_Lab_CSL",
    "settings": {
        "appearence_mode": "light"
    },
    "fonts": {
        "my_font_1": {
            #"family":  "Minimalust",
            "family": 'Courier New',
            "size": 13,
            "weight": "normal"
        }
    },
    "colors": {
        "MEDIUMGREEN_FG": "#218530",
        "MEDIUMGREEN_HOVER_FG": "#319941",
        "LIGHTRED_FG": "#c94259",
        "LIGHTRED_HOVER_FG": "#d9596e",
        "SKYBLUE_FG": "#E3E4FA",
        #"SKYBLUE_FG": "#e8e3fa",
        "GREY_HOVER_FG": "#a3a2a3",
        "GREY_FG": "#6b6a6b",
        "LIGHT_YELLOW_FG": "#fff1cc"
    },

    "algorithm_properties": {
        'RF': {
            'hp_optim': {
                'method': {
                    'opts': _COMMON_PROPS['hp_optim']['method_opts_with_Optuna']
                },
                'scoring': {
                    'opts': _COMMON_PROPS['hp_optim']['scoring_opts']
                },
                'cvfolds': {'min': 2, 'max': 20, 'default': 2 },
                'n_estimators': {
                    'min_val': 50, 'max_val':200, 'max_steps': 10, 'default_from': 90, 'default_to': 100, 'default_max_steps': 10
                },
                'criterion': {
                    'options': ['gini', 'entropy', 'log_loss'], 
                    'min_num_of_choices': 2
                },
                'max_depth': { 'min_val': 1, 'max_val':30, 'default_from': 11, 'default_to':12 },
                'min_samples_split': { 'min_val': 2, 'max_val':10, 'default_from': 2, 'default_to':3},
                'min_samples_leaf': { 'min_val': 1, 'max_val':10, 'default_from': 1, 'default_to':2 },
                # 'cvfolds': {'min': 3, 'max': 20, 'default': 5 },
                # 'n_estimators': {
                #     'min_val': 50, 'max_val':200, 'max_steps': 10, 'default_from': 50, 'default_to': 100, 'default_max_steps': 5
                # },
                # 'criterion': {
                #     'options': ['gini', 'entropy', 'log_loss'], 
                #     'min_num_of_choices': 2
                # },
                # 'max_depth': { 'min_val': 1, 'max_val':30, 'default_from': 1, 'default_to':5 },
                # 'min_samples_split': { 'min_val': 2, 'max_val':10, 'default_from': 2, 'default_to':5},
                # 'min_samples_leaf': { 'min_val': 1, 'max_val':10, 'default_from': 1, 'default_to':5 },
            },
            'model_build': {
                'n_estimators': {'default_val':100},
                'criterion': {'default_idx':0, 'options': ['gini', 'entropy', 'log_loss']},
                'max_depth': {'default_val': None},
                'min_samples_split': {'default_val':2},
                'min_samples_leaf': {'default_val':1},
                'min_impurity_decrease': {'default_val':0.0},
                'random_state': {'default_val':None},
                'warm_start': {'default_idx':1, 'options': ['True', 'False']},
                'max_features': {'default_val':'sqrt'},
                'min_weight_fraction_leaf':{'default_val':0.0},
                'max_leaf_nodes':{'default_val':None}
            }
        },
        'SVM': {
            'hp_optim': {
                'method': {
                    'opts': _COMMON_PROPS['hp_optim']['method_opts_with_Optuna']
                },
                'scoring': {
                    'opts': _COMMON_PROPS['hp_optim']['scoring_opts']
                },
                'cvfolds': {'min': 3, 'max': 20, 'default': 5 },
                'C': {
                    'options':[f'{10**x}' for x in range(-1,4)],
                    'min_num_of_choices': 2
                },
                'kernel':{
                    'options':['linear','poly','rbf', 'sigmoid'],
                    'min_num_of_choices': 2
                },
                'gamma':{
                    'options':['scale','auto'],
                    'min_num_of_choices': 2
                }
            },
            'model_build': {
                'C':{'default_val':1.0},
                'kernel':{'default_idx':2, 'options': ['linear','poly','rbf', 'sigmoid']},
                'degree':{'default_val':3},
                'gamma':{'default_idx':0, 'options': ['scale','auto']},
                'coef0':{'default_val':0.0},
                'tol':{'default_val':1e-3},
                'shrinking':{'default_idx':0, 'options': ['True','False']},
                'probability':{'default_idx':0, 'options': ['True','False']},
                'random_state':{'default_val':None},
                'break_ties':{'default_idx':1, 'options': ['True','False']},
                'decision_function_shape':{'default_idx':0, 'options': ['ovr','ovo']},
            }
        },
        'LDA': {
            'hp_optim':{
                'method': {
                    'opts': _COMMON_PROPS['hp_optim']['method_opts_with_Optuna']
                },
                'scoring': {
                    'opts': _COMMON_PROPS['hp_optim']['scoring_opts']
                },
                'cvfolds': {'min': 3, 'max': 20, 'default': 5 },
                'solver': {
                    'options':['svd', 'lsqr', 'eigen'],
                    'min_num_of_choices': 2
                },
            },
            'model_build':{
                'solver':{'default_idx':0, 'options': ['svd', 'lsqr', 'eigen']},
                'shrinkage':{'default_val':None},
                'n_components':{'default_val':None},
                'tol':{'default_val':1e-3},
                'store_covariance':{'default_idx':1, 'options': ['True','False']},
            }
        },
        'LR': {
            'hp_optim': {
                'method': {
                    'opts': _COMMON_PROPS['hp_optim']['method_opts_with_Optuna']
                },
                'scoring': {
                    'opts': _COMMON_PROPS['hp_optim']['scoring_opts']
                },
                'cvfolds': {'min': 3, 'max': 20, 'default': 5 },
                'C': {
                    'options': [f'{10**x}' for x in range(-1,2)], 
                    'min_num_of_choices': 2
                },
                'penalty': {
                    'options': ['l1','l2','elasticnet', 'None'], 
                    'min_num_of_choices': 2
                },
                'solver': {
                    'options': ['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga'], 
                    'min_num_of_choices': 2
                },
            },
            'model_build':{
                'l1_ratio':{'default_val':None},
                'penalty': {'default_idx':1, 'options': ['l1','l2','elasticnet', 'None']},
                'tol': {'default_val':1e-4},
                'C': {'default_val':1.0},
                'fit_intercept': {'default_idx':0, 'options': ['True', 'False']},
                'intercept_scaling': {'default_val':1.0},
                'random_state': {'default_val':None},
                'solver': {'default_idx':0, 'options': ['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga']},
                'warm_start': {'default_idx':1, 'options': ['True', 'False']},
                'max_iter': {'default_val':100},
                'multi_class':{'default_idx':0, 'options': ['auto', 'ovr', 'multinomial']},
                'n_jobs': {'default_val':None},
            }
        },
        'KNN':{
            'hp_optim': {
                'method': {
                    'opts': _COMMON_PROPS['hp_optim']['method_opts_with_Optuna']
                },
                'scoring': {
                    'opts': _COMMON_PROPS['hp_optim']['scoring_opts']
                },
                'cvfolds': {'min': 3, 'max': 20, 'default': 5 },
                'n_neighbors': { 'min_val': 2, 'max_val':20, 'default_from': 2, 'default_to':10 },
                'leaf_size': {'min_val': 10, 'max_val':50, 'default_from': 10, 'default_to':50 },
                'p': {'min_val': 1, 'max_val':5, 'default_from': 1, 'default_to':2},
            },
            'model_build': {
                'n_neighbors': {'default_val':5 },
                'weights': {'default_idx':0, 'options': ['uniform', 'distance'] },
                'p': {'default_val':2.0 },
                'leaf_size': {'default_val':30 },
                'algorithm': {'default_idx':0, 'options': ['auto', 'ball_tree', 'kd_tree', 'brute'] },
                'n_jobs': {'default_val':None },
            }
        },
        'GB': {
            'hp_optim': {
                'method': {
                    'opts': _COMMON_PROPS['hp_optim']['method_opts_with_Optuna']
                },
                'scoring': {
                    'opts': _COMMON_PROPS['hp_optim']['scoring_opts']
                },
                'cvfolds': {'min': 3, 'max': 20, 'default': 5 },
                'n_estimators': {
                    'min_val': 10, 'max_val':200, 'max_steps': 10, 'default_from': 10, 'default_to': 200, 'default_max_steps': 10
                },
                'criterion': {
                    'options': ['friedman_mse', 'squared_error'], 
                    'min_num_of_choices': 2
                },
                'learning_rate': { 'min_val': 1e-3, 'max_val':1, 'default_from': 1e-2, 'default_to':1e-1 },
                'max_depth': {'min_val': 2, 'max_val':32, 'default_from': 2, 'default_to':10 },
                'min_samples_split': { 'min_val': 2, 'max_val':10, 'default_from': 2, 'default_to':5},
                'min_samples_leaf': { 'min_val': 1, 'max_val':10, 'default_from': 1, 'default_to':5 },
            },
            'model_build': {
                'loss': {'default_idx':0, 'options': ['log_loss', 'exponential']},
                'learning_rate': { 'default_val':0.1 },
                'n_estimators': { 'default_val':100 },
                'subsample': { 'default_val':0.1 },
                'criterion': {'default_idx':0, 'options': ['friedman_mse', 'squared_error']},
                'min_samples_split': {'default_val':2},
                'min_samples_leaf': {'default_val':1},
                'min_weight_fraction_leaf': {'default_val':0.0},
                'max_depth': {'default_val':3},
                'min_impurity_decrease': {'default_val':0.0},
                'random_state': {'default_val':None},
                'warm_start': {'default_idx':1, 'options': ['True', 'False']},
                'max_features': {'default_val':None},
                'validation_fraction': { 'default_val':0.1 },
                'n_iter_no_change': {'default_val':None},
                'tol': { 'default_val':1e-4 },
            }
        },
        'MLP': {
            'hp_optim': {
                'method': {
                    'opts': _COMMON_PROPS['hp_optim']['method_opts_with_Optuna']
                },
                'scoring': {
                    'opts': _COMMON_PROPS['hp_optim']['scoring_opts']
                },
                'cvfolds': {'min': 2, 'max': 20, 'default': 2 },
                'hidden_layer_size': {
                    'min_val': 50, 'max_val':300, 'max_steps': 50, 'default_from': 64, 'default_to': 65, 'default_max_steps': 1
                },
                'num_of_hidden_layers': {
                    'min_val': 2, 'max_val':10, 'default_from': 2, 'default_to':4
                },
                'activation': {
                    'options': ['identity', 'logistic', 'tanh', 'relu'], 
                    'min_num_of_choices': 2
                },
                'solver': {
                    'options': ['lbfgs', 'sgd', 'adam'], 
                    'min_num_of_choices': 2
                },
                'alpha': {
                    'options':[f'{10**x}' for x in range(-4,1)],
                    'min_num_of_choices': 2
                },
                'learning_rate': {
                    'options': ['constant', 'invscaling', 'adaptive'], 
                    'min_num_of_choices': 2
                },
            },
            'model_build': {
                'hidden_layer_sizes': {'default_val':'(100,)'},
                'activation': {
                    'default_idx':3, 
                    'options': ['identity', 'logistic', 'tanh', 'relu']
                },
                'solver': {
                    'default_idx':2, 
                    'options': ['lbfgs', 'sgd', 'adam']
                },
                'alpha': {'default_val': 0.0001},
                'batch_size': {'default_val': 'auto'},
                'learning_rate': {
                    'default_idx':0, 
                    'options': ['constant', 'invscaling', 'adaptive']
                },
                'learning_rate_init': {'default_val': 0.001},
                'power_t': {'default_val': 0.5},
                'max_iter': {'default_val': 200},
                'shuffle': {'default_idx':0, 'options': ['True', 'False']},
                'random_state': {'default_val':None},
                'tol': {'default_val': 1e-4},
                'verbose': {'default_idx':1, 'options': ['True', 'False']},
                'warm_start': {'default_idx':1, 'options': ['True', 'False']},
                'momentum': {'default_val': 0.9},
                'nesterovs_momentum': {'default_idx':0, 'options': ['True', 'False']},
                'early_stopping': {'default_idx':1, 'options': ['True', 'False']},
                'validation_fraction': {'default_val': 0.1},
                'n_iter_no_change': {'default_val': 10},
                'max_fun': {'default_val': 15000}
            }
        },
    },

    'plot_properties' : {
        'ROC_CURVE': {
            'COLOR': 'red',
            'LW': 3
        },
        'DIAGONAL_REF_LINE': {
            'COLOR': 'blue'
        },
        'TITLE' : {
            'FONT_SIZE': 21,
            'FONT_WEIGHT': "bold",
            'FONT_STYLE' : "Times New Roman"
        },
        'RC_PARAMS': {
            'FONT_SIZE': 12,
            'FONT_WEIGHT': "bold",
            'FONT_STYLE' : "Courier New"
        },
        'SHAP': {
            'ENABLED': True,
            'CLASS_INDEX': 1, # can be 0/1 or even 2 if n_classes>=2
            'COLOR_SCHEME': 'turbo_r'
        },
        'OPTUNA': {
            'plot1': {
                'layout' : {
                    'title': 'Optimization History',
                    'height': 600,
                    'width': 800,
                    'plot_bgcolor': 'whitesmoke',
                    'paper_bgcolor': 'white',
                    'xaxis': {
                        'tickangle':45,
                        'showgrid':True,
                        'gridcolor':'white',
                        'dtick':50, 
                        'zeroline':False,
                        'zerolinecolor':'white',
                        'zerolinewidth':2
                    },
                    'yaxis': {
                        'tickangle':45,
                        'showgrid':True,
                        'gridcolor':'white',
                        'dtick':50, 
                        'zeroline':False,
                        'zerolinecolor':'white',
                        'zerolinewidth':2
                    }
                },
                'traces': {
                    'marker_color':'blue',
                    'line_color': 'red',
                }
            },
            'plot2': {
                'layout': {
                    'title': 'Hyperparameter Importances',
                    'height': 500,
                    'width': 800,
                    'plot_bgcolor': 'whitesmoke',
                    'paper_bgcolor': 'white',
                    'bargap':0.2,
                    'xaxis': {
                        'title': 'Importance',
                        'showline':True,
                        'linecolor': 'white',
                        'linewidth':1,
                        'tickangle':45,
                        'ticklen':15,
                        'dtick':0.1,
                        'showgrid':True,
                        'gridcolor':'white',
                        'gridwidth':0.5,
                    },
                    'yaxis': {
                        'title': 'HyperParameters',
                        'tickangle':0,
                    }
                },
                'traces': {
                    'marker_color':'red',
                }
            },
            'plot3': {
                'layout': {
                    'title': 'Parallel Coordinate Plot',
                    'height': 600,
                    'width': 800,
                    'plot_bgcolor': 'whitesmoke',
                    'paper_bgcolor': 'white',
                },
                'traces': {
                    'line': {
                        'colorscale': 'turbo_r',
                        'showscale':True,
                        'colorbar': {
                            'title': 'ColorScale'
                        }
                    }
                }
            },
            'plot4': {
                'layout': {
                    'title':'Slice Plot',
                    'height': 600,
                    'width': 1000,
                    'plot_bgcolor': 'whitesmoke',
                    'paper_bgcolor': 'white',
                },
                'traces': {
                    'marker':{
                        'size':6,
                        'colorbar':{ 
                            'title': 'Color_Scale'
                        },
                        'colorscale':'portland'
                    }
                },
                'xaxis': {
                    'showgrid':True,
                    'gridcolor':'white',
                    'gridwidth':1,
                    'zeroline':True,
                    'zerolinecolor':'white',
                    'showticklabels':True
                },
                'yaxis': {
                    'showgrid':True,
                    'gridcolor':'white',
                    'gridwidth':1,
                    'zeroline':True,
                    'zerolinecolor':'white',
                    'showticklabels':True
                }
            },
            'plot5': {
                'layout': {
                    'title':'Contour Plot',
                    'height': 1000,
                    'width': 1000,
                    'plot_bgcolor': 'whitesmoke',
                    'paper_bgcolor': 'white',
                },
                'xaxis': {
                    'showline':True,
                    'linecolor': 'white',
                    'linewidth':1,
                    'showgrid':True,
                    'gridcolor':'white',
                    'gridwidth':1,
                },
                'yaxis': {
                    'showline':True,
                    'linecolor': 'white',
                    'linewidth':1,
                    'showgrid':True,
                    'gridcolor':'white',
                    'gridwidth':1,
                },
                'traces': {
                    'selector': {
                        'type': 'contour'
                    },
                    'colorscale': 'portland',
                    'contours': {
                        'coloring':'fill',
                        'showlines':True,  # Show contour lines
                        'size':0.5,
                        #'start':0,
                        #'end':1
                    },
                    'line': {
                        'color':'white',
                        'width':1
                    },
                    'colorbar': {
                        'title':'Color_Scale',
                        'titleside': 'right',
                        'tickfont': {
                            'size':12,
                            'color': 'black'
                        }
                    }
                }
            },
        }
    }
}