import os
#import sys
from threading import Thread, Event
import customtkinter as ctk
from PIL import Image, ImageTk
from constants import my_config_manager, IMG_DIR
from util.services import *
from util.gui.panels import *
from util.gui.labelframes import HyperParamOptim_AlgoLabelFrame, ModelBuild_AlgoLabelFrame, MyRangeEntryField, MultiSelectEntryField, MyStepRangeEntryField, MyLogarithmicRangeEntryField,StringField, OptionMenuField

#sys.path.append(os.path.abspath(os.path.dirname(__file__)))
ctk.set_appearance_mode(
    mode_string = my_config_manager.get("settings.appearence_mode")
)

root=ctk.CTk()
root.title(my_config_manager.get("app_name"))
#root.resizable(False, False)
root.focus_force()
root.grid_columnconfigure(tuple(range(1,8)), weight=1) # 8 columns
root.grid_rowconfigure(tuple(range(2,11)),weight=1) # Only Side_panel and task_panel will expand

# FONTS
FONT_1_DATA = my_config_manager.get('fonts.1')

MY_FONT_1 = ctk.CTkFont(
    family=FONT_1_DATA["family"], 
    size=FONT_1_DATA["size"], 
    weight=FONT_1_DATA["weight"], 
)
# IMAGES PATH
RESULTS_LOADING_IMG_PATH = os.path.join(IMG_DIR, 'loading.gif')
HP_OPTIMIZATION_ONGOING_IMG_PATH = os.path.join(IMG_DIR, 'optimization.gif')

UPLOAD_IMG = {
    'PATH': os.path.join(IMG_DIR, 'upload.png'), 'SIZE': (20,20)
}
D3_LAB_LOGO = {
    'PATH': os.path.join(IMG_DIR, 'banner.png'), 'SIZE': (75,75)
}
DEFAULT_BANNER_IMG = {
    'PATH': os.path.join(IMG_DIR, 'default_panel.png'), 'SIZE': (637,470)
}
HP_OPTIM_IMG = {
    'PATH': os.path.join(IMG_DIR, 'hp_optim.png'), 'SIZE': (40,40)
}
MODEL_BUILD_IMG = {
    'PATH': os.path.join(IMG_DIR, 'model_build.png'), 'SIZE': (40,40)
}
DATASET_DIV_IMG = {
    'PATH': os.path.join(IMG_DIR, 'dataset_division.png'), 'SIZE': (40,40)
}
SETTINGS_IMG = {
    'PATH': os.path.join(IMG_DIR, 'settings.png'), 'SIZE': (30,30)
}


# GLOBAL VARIABLES
ALL_LOADED_FEATURES = ctk.StringVar()
SELECTED_FEATURES = ctk.StringVar()
TRAIN_FILE_PATH = ctk.StringVar()
TEST_FILE_PATH = ctk.StringVar()
COLORS = my_config_manager.get('colors')
ALGO_MAP = {
    'RF': {
        'algo_name': 'Random Forest', 
        'algo_hp_optim_func': RF_HP_OPTIM_SUBMIT, 
        'algo_model_build_func': RF_MODEL_BUILD_SUBMIT
    },
    'SVM': {
        'algo_name': 'Support Vector Machine', 
        'algo_hp_optim_func': SVM_HP_OPTIM_SUBMIT, 
        'algo_model_build_func': SVM_MODEL_BUILD_SUBMIT
    },
    'LR': {
        'algo_name': 'Logistic Regression', 
        'algo_hp_optim_func': LR_HP_OPTIM_SUBMIT, 
        'algo_model_build_func': LR_MODEL_BUILD_SUBMIT
    },
    'LDA': {
        'algo_name': 'Linear Discriminant Analysis', 
        'algo_hp_optim_func': LDA_HP_OPTIM_SUBMIT, 
        'algo_model_build_func': LDA_MODEL_BUILD_SUBMIT
    },
    'KNN': {
        'algo_name': 'K-Nearest Neighbors', 
        'algo_hp_optim_func': KNN_HP_OPTIM_SUBMIT, 
        'algo_model_build_func': KNN_MODEL_BUILD_SUBMIT
    },
    'GB': {
        'algo_name': 'Gradient Boosting', 
        'algo_hp_optim_func': GB_HP_OPTIM_SUBMIT, 
        'algo_model_build_func': GB_MODEL_BUILD_SUBMIT
    },
    'MLP': {
        'algo_name': 'MultiLayer Perceptron', 
        'algo_hp_optim_func': MLP_HP_OPTIM_SUBMIT, 
        'algo_model_build_func': MLP_MODEL_BUILD_SUBMIT
    },
}
HP_OPTIM_SELECTED_ALGORITHM = ctk.StringVar()
MODEL_BUILD_SELECTED_ALGORITHM = ctk.StringVar()

icon_img = Image.open(D3_LAB_LOGO['PATH'])
icon_photo = ImageTk.PhotoImage(icon_img)
root.wm_iconphoto(True, icon_photo)

# DATASET FRAME
DataSetPanel(
    master=root, 
    LOADED_FEATURES=ALL_LOADED_FEATURES, 
    SELECTED_FEATURES=SELECTED_FEATURES, 
    train_entryVar=TRAIN_FILE_PATH, 
    test_entryVar=TEST_FILE_PATH,
    my_font=MY_FONT_1,
    colors_fgAndBtns={
        'fg': COLORS['SKYBLUE_FG'], 
        'btn': {'fg': COLORS['MEDIUMGREEN_FG'], 'hover': COLORS['MEDIUMGREEN_HOVER_FG']}
    },
    img_pathsAndSizes={
        'logo': {'path': D3_LAB_LOGO['PATH'], 'size': D3_LAB_LOGO['SIZE']}, 
        'upload': {'path': UPLOAD_IMG['PATH'], 'size': UPLOAD_IMG['SIZE']}
    }
)

side_panel = SidePanel(
    master=root,
    my_font=MY_FONT_1,
    colors_fgAndBtns= {
        'fg': COLORS['SKYBLUE_FG'], 
        'btns': {
            'hp_optim': {'fg': COLORS['LIGHTRED_FG'], 'hover': COLORS['LIGHTRED_HOVER_FG']}, 
            'model_build': { 'fg': COLORS['MEDIUMGREEN_FG'], 'hover': COLORS['MEDIUMGREEN_HOVER_FG']},
            'dataset_div': { 'fg': COLORS['ORANGE_FG'], 'hover': COLORS['ORANGE_HOVER_FG']},
            'settings': { 'fg': COLORS['SLATEBLUE_FG'], 'hover': COLORS['SLATEBLUE_HOVER_FG']}
        }
    },
    img_pathsAndSizes={
        'hp_optim': {'path': HP_OPTIM_IMG['PATH'], 'size': HP_OPTIM_IMG['SIZE']}, 
        'model_build': {'path': MODEL_BUILD_IMG['PATH'], 'size': MODEL_BUILD_IMG['SIZE']},
        'dataset_div': {'path': DATASET_DIV_IMG['PATH'], 'size': DATASET_DIV_IMG['SIZE']},
        'settings': {'path': SETTINGS_IMG['PATH'], 'size': SETTINGS_IMG['SIZE']},
    }
)

taskSelectBtns = side_panel.GET_SELECT_TASK_BTNS()
task_panel = TaskPanel(
    master=root,
    my_font=MY_FONT_1,
    fg_color=COLORS['SKYBLUE_FG'], 
    taskSelectBtns_fromSidePanel=taskSelectBtns
)
taskPanelMap = task_panel._get_task_panels()
# {'hp_optim': hyperparam_optim_panel, 'model_build': model_build_panel, 'default': default_panel}

# DATASET DIVISION PANEL ----------------------
dataset_div_panel = taskPanelMap['dataset_div']
DataSetDivFrame(
    masterFrame=dataset_div_panel,
    my_font=MY_FONT_1,
    colors={
        'fg': COLORS['SKYBLUE_FG'], 
        'btn': {'fg': COLORS['MEDIUMGREEN_FG'], 'hover': COLORS['MEDIUMGREEN_HOVER_FG']},
        'optionmenu': {'fg': COLORS['GREY_FG'], 'hover': COLORS['GREY_HOVER_FG']},
    },
    img_pathsAndSizes={ 
        'upload': {'path': UPLOAD_IMG['PATH'], 'size': UPLOAD_IMG['SIZE']}
    }
)

# Default Panel
default_panel = taskPanelMap['default']
default_panel.grid_rowconfigure(0, weight=1)
default_panel.grid_columnconfigure(0, weight=1)
default_panel_image = ctk.CTkImage(light_image=Image.open(DEFAULT_BANNER_IMG['PATH']), size=DEFAULT_BANNER_IMG['SIZE'])
default_panel_label = ctk.CTkLabel(default_panel, image=default_panel_image, text="")
default_panel_label.grid(row=0, column=0, sticky=ctk.NSEW)

# HP_OPTIM_PANEL------------------------------------------------------------------------------------------
hyperparam_optim_panel = taskPanelMap['hp_optim']

hp_optim_featureAndAlgoSelectFrame = FeatureAndAlgorithmFrame(
    masterFrame=hyperparam_optim_panel,
    my_font=MY_FONT_1,
    colors={
        'fg':COLORS['SKYBLUE_FG'], 
        'btn':{'fg':COLORS['GREY_FG'], 'hover':COLORS['GREY_FG']}
    },
    LOADED_FEATURES=ALL_LOADED_FEATURES,
    SELECTED_FEATURES=SELECTED_FEATURES,
    LIST_OF_ALGORITHMS=[algoVal['algo_name'] for algoVal in ALGO_MAP.values()],
    SELECTED_ALGORITHM=HP_OPTIM_SELECTED_ALGORITHM,
    TRAIN_FILE_PATH=TRAIN_FILE_PATH,
    TEST_FILE_PATH=TEST_FILE_PATH
)

RF_HP_OPTIM_DATA = my_config_manager.get('algorithm_properties.RF.hp_optim')
RF_HP_OPTIM_DATA_AND_VARS = {
    'method_opts': RF_HP_OPTIM_DATA['method']['opts'],
    'method_selected_optVar': ctk.StringVar(value=RF_HP_OPTIM_DATA['method']['opts'][0]),
    'scoring_opts': RF_HP_OPTIM_DATA['scoring']['opts'],
    'scoring_selected_optVar': ctk.StringVar(value=RF_HP_OPTIM_DATA['scoring']['opts'][0]),
    'cvfolds_entryVar': ctk.IntVar(value=RF_HP_OPTIM_DATA['cvfolds']['default']),
    'cvfolds_min': RF_HP_OPTIM_DATA['cvfolds']['min'],
    'cvfolds_max': RF_HP_OPTIM_DATA['cvfolds']['max'],

    'hp_optim_field_configs': {
        'n_estimators': {
            'type': MyStepRangeEntryField(
                from_var=ctk.IntVar(value=RF_HP_OPTIM_DATA['n_estimators']['default_from']),
                to_var=ctk.IntVar(value=RF_HP_OPTIM_DATA['n_estimators']['default_to']),
                step_var=ctk.IntVar(value=RF_HP_OPTIM_DATA['n_estimators']['default_max_steps']),
                min_val=RF_HP_OPTIM_DATA['n_estimators']['min_val'],
                max_val=RF_HP_OPTIM_DATA['n_estimators']['max_val'],
                max_steps=RF_HP_OPTIM_DATA['n_estimators']['max_steps']
            ),
            'grid': {'row':0,'col':0,'colspan':2}
        },
        'criterion': {
            'type': MultiSelectEntryField(
                options=RF_HP_OPTIM_DATA['criterion']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(RF_HP_OPTIM_DATA['criterion']['options']))
            ),
            'grid': {'row':1,'col':0,'colspan':1}
        },
        'max_depth': {
            'type': MyRangeEntryField(
                from_var=ctk.IntVar(value=RF_HP_OPTIM_DATA['max_depth']['default_from']),
                to_var=ctk.IntVar(value=RF_HP_OPTIM_DATA['max_depth']['default_to']),
                min_val=RF_HP_OPTIM_DATA['max_depth']['min_val'],
                max_val=RF_HP_OPTIM_DATA['max_depth']['max_val'],
            ),
            'grid': {'row':1,'col':1,'colspan':1}
        },
        'min_samples_split': {
            'type': MyRangeEntryField(
                from_var=ctk.IntVar(value=RF_HP_OPTIM_DATA['min_samples_split']['default_from']),
                to_var=ctk.IntVar(value=RF_HP_OPTIM_DATA['min_samples_split']['default_to']),
                min_val=RF_HP_OPTIM_DATA['min_samples_split']['min_val'],
                max_val=RF_HP_OPTIM_DATA['min_samples_split']['max_val'],
            ),
            'grid': {'row':2,'col':0,'colspan':1}
        },
        'min_samples_leaf': {
            'type': MyRangeEntryField(
                from_var=ctk.IntVar(value=RF_HP_OPTIM_DATA['min_samples_leaf']['default_from']),
                to_var=ctk.IntVar(value=RF_HP_OPTIM_DATA['min_samples_leaf']['default_to']),
                min_val=RF_HP_OPTIM_DATA['min_samples_leaf']['min_val'],
                max_val=RF_HP_OPTIM_DATA['min_samples_leaf']['max_val'],
            ),
            'grid': {'row':2,'col':1,'colspan':1}
        }
    }
}
RF_hp_optim_panel = HyperParamOptim_AlgoLabelFrame(
    master_panel=hyperparam_optim_panel,
    algoMap=ALGO_MAP,
    algo_name='RF',
    algo_hp_optim_fields_data=RF_HP_OPTIM_DATA_AND_VARS,
    my_font=MY_FONT_1,
    fg_color=COLORS['SKYBLUE_FG'],
    result_Loading_ImgPath=HP_OPTIMIZATION_ONGOING_IMG_PATH,
    trainVar=TRAIN_FILE_PATH,
    testVar=TEST_FILE_PATH,
    SELECTED_FEATURES=SELECTED_FEATURES,

    hyperParamsFrame_NumOfCells=2
)._GET_ALGO_LABELFRAME()

#SVM
SVM_HP_OPTIM_DATA = my_config_manager.get('algorithm_properties.SVM.hp_optim')
SVM_HP_OPTIM_DATA_AND_VARS = {
    'method_opts': SVM_HP_OPTIM_DATA['method']['opts'],
    'method_selected_optVar': ctk.StringVar(value=SVM_HP_OPTIM_DATA['method']['opts'][0]),
    'scoring_opts': SVM_HP_OPTIM_DATA['scoring']['opts'],
    'scoring_selected_optVar': ctk.StringVar(value=SVM_HP_OPTIM_DATA['scoring']['opts'][0]),
    'cvfolds_entryVar': ctk.IntVar(value=SVM_HP_OPTIM_DATA['cvfolds']['default']),
    'cvfolds_min': SVM_HP_OPTIM_DATA['cvfolds']['min'],
    'cvfolds_max': SVM_HP_OPTIM_DATA['cvfolds']['max'],

    'hp_optim_field_configs': {
        'C': {
            'type': MultiSelectEntryField(
                options=SVM_HP_OPTIM_DATA['C']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(list(map(str, SVM_HP_OPTIM_DATA['C']['options']))))
            ),
            'grid': {'row':0,'col':0,'colspan':1}
        },
        'kernel': {
            'type': MultiSelectEntryField(
                options=SVM_HP_OPTIM_DATA['kernel']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(SVM_HP_OPTIM_DATA['kernel']['options']))
            ),
            'grid': {'row':1,'col':0,'colspan':2}
        },
        'gamma': {
            'type': MultiSelectEntryField(
                options=SVM_HP_OPTIM_DATA['gamma']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(SVM_HP_OPTIM_DATA['gamma']['options']))
            ),
            'grid': {'row':0,'col':1,'colspan':1}
        },
    }
}
SVM_hp_optim_panel = HyperParamOptim_AlgoLabelFrame(
    master_panel=hyperparam_optim_panel,
    algoMap=ALGO_MAP,
    algo_name='SVM',
    algo_hp_optim_fields_data=SVM_HP_OPTIM_DATA_AND_VARS,
    my_font=MY_FONT_1,
    fg_color=COLORS['SKYBLUE_FG'],
    result_Loading_ImgPath=HP_OPTIMIZATION_ONGOING_IMG_PATH,
    SELECTED_FEATURES=SELECTED_FEATURES,
    trainVar=TRAIN_FILE_PATH,
    testVar=TEST_FILE_PATH,
    hyperParamsFrame_NumOfCells=2
)._GET_ALGO_LABELFRAME()

#LDA
LDA_HP_OPTIM_DATA = my_config_manager.get('algorithm_properties.LDA.hp_optim')
LDA_HP_OPTIM_DATA_AND_VARS = {
    'method_opts': LDA_HP_OPTIM_DATA['method']['opts'],
    'method_selected_optVar': ctk.StringVar(value=LDA_HP_OPTIM_DATA['method']['opts'][0]),
    'scoring_opts': LDA_HP_OPTIM_DATA['scoring']['opts'],
    'scoring_selected_optVar': ctk.StringVar(value=LDA_HP_OPTIM_DATA['scoring']['opts'][0]),
    'cvfolds_entryVar': ctk.IntVar(value=LDA_HP_OPTIM_DATA['cvfolds']['default']),
    'cvfolds_min': LDA_HP_OPTIM_DATA['cvfolds']['min'],
    'cvfolds_max': LDA_HP_OPTIM_DATA['cvfolds']['max'],

    'hp_optim_field_configs': {
        'solver': {
            'type': MultiSelectEntryField(
                options=LDA_HP_OPTIM_DATA['solver']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(LDA_HP_OPTIM_DATA['solver']['options']))
            ),
            'grid': {'row':0,'col':0,'colspan':1}
        }
    }
}
LDA_hp_optim_panel = HyperParamOptim_AlgoLabelFrame(
    master_panel=hyperparam_optim_panel,
    algoMap=ALGO_MAP,
    algo_name='LDA',
    algo_hp_optim_fields_data=LDA_HP_OPTIM_DATA_AND_VARS,
    my_font=MY_FONT_1,
    fg_color=COLORS['SKYBLUE_FG'],
    result_Loading_ImgPath=HP_OPTIMIZATION_ONGOING_IMG_PATH,
    SELECTED_FEATURES=SELECTED_FEATURES,
    trainVar=TRAIN_FILE_PATH,
    testVar=TEST_FILE_PATH,
    hyperParamsFrame_NumOfCells=1
)._GET_ALGO_LABELFRAME()

#LR
LR_HP_OPTIM_DATA = my_config_manager.get('algorithm_properties.LR.hp_optim')
LR_HP_OPTIM_DATA_AND_VARS = {
    'method_opts': LR_HP_OPTIM_DATA['method']['opts'],
    'method_selected_optVar': ctk.StringVar(value=LR_HP_OPTIM_DATA['method']['opts'][0]),
    'scoring_opts': LR_HP_OPTIM_DATA['scoring']['opts'],
    'scoring_selected_optVar': ctk.StringVar(value=LR_HP_OPTIM_DATA['scoring']['opts'][0]),
    'cvfolds_entryVar': ctk.IntVar(value=LR_HP_OPTIM_DATA['cvfolds']['default']),
    'cvfolds_min': LR_HP_OPTIM_DATA['cvfolds']['min'],
    'cvfolds_max': LR_HP_OPTIM_DATA['cvfolds']['max'],

    'hp_optim_field_configs': {
        'C': {
            'type': MultiSelectEntryField(
                options=LR_HP_OPTIM_DATA['C']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(list(map(str, LR_HP_OPTIM_DATA['C']['options']))))
            ),
            'grid': {'row':0,'col':0,'colspan':1}
        },
        'penalty': {
            'type': MultiSelectEntryField(
                options=LR_HP_OPTIM_DATA['penalty']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(LR_HP_OPTIM_DATA['penalty']['options']))
            ),
            'grid': {'row':0,'col':1,'colspan':1}
        },
        'solver': {
            'type': MultiSelectEntryField(
                options=LR_HP_OPTIM_DATA['solver']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(LR_HP_OPTIM_DATA['solver']['options']))
            ),
            'grid': {'row':2,'col':0,'colspan':2}
        },
    }
}
LR_hp_optim_panel = HyperParamOptim_AlgoLabelFrame(
    master_panel=hyperparam_optim_panel,
    algoMap=ALGO_MAP,
    algo_name='LR',
    algo_hp_optim_fields_data=LR_HP_OPTIM_DATA_AND_VARS,
    my_font=MY_FONT_1,
    fg_color=COLORS['SKYBLUE_FG'],
    result_Loading_ImgPath=HP_OPTIMIZATION_ONGOING_IMG_PATH,
    SELECTED_FEATURES=SELECTED_FEATURES,
    trainVar=TRAIN_FILE_PATH,
    testVar=TEST_FILE_PATH,
    hyperParamsFrame_NumOfCells=2
)._GET_ALGO_LABELFRAME()

#KNN
KNN_HP_OPTIM_DATA = my_config_manager.get('algorithm_properties.KNN.hp_optim')
KNN_HP_OPTIM_DATA_AND_VARS = {
    'method_opts': KNN_HP_OPTIM_DATA['method']['opts'],
    'method_selected_optVar': ctk.StringVar(value=KNN_HP_OPTIM_DATA['method']['opts'][0]),
    'scoring_opts': KNN_HP_OPTIM_DATA['scoring']['opts'],
    'scoring_selected_optVar': ctk.StringVar(value=KNN_HP_OPTIM_DATA['scoring']['opts'][0]),
    'cvfolds_entryVar': ctk.IntVar(value=KNN_HP_OPTIM_DATA['cvfolds']['default']),
    'cvfolds_min': KNN_HP_OPTIM_DATA['cvfolds']['min'],
    'cvfolds_max': KNN_HP_OPTIM_DATA['cvfolds']['max'],

    'hp_optim_field_configs': {
        'n_neighbors': {
            'type': MyRangeEntryField(
                from_var=ctk.IntVar(value=KNN_HP_OPTIM_DATA['n_neighbors']['default_from']),
                to_var=ctk.IntVar(value=KNN_HP_OPTIM_DATA['n_neighbors']['default_to']),
                min_val=KNN_HP_OPTIM_DATA['n_neighbors']['min_val'],
                max_val=KNN_HP_OPTIM_DATA['n_neighbors']['max_val'],
            ),
            'grid': {'row':0,'col':0,'colspan':1}
        },
        'leaf_size': {
            'type': MyRangeEntryField(
                from_var=ctk.IntVar(value=KNN_HP_OPTIM_DATA['leaf_size']['default_from']),
                to_var=ctk.IntVar(value=KNN_HP_OPTIM_DATA['leaf_size']['default_to']),
                min_val=KNN_HP_OPTIM_DATA['leaf_size']['min_val'],
                max_val=KNN_HP_OPTIM_DATA['leaf_size']['max_val'],
            ),
            'grid': {'row':1,'col':0,'colspan':1}
        },
        'p': {
            'type': MyRangeEntryField(
                from_var=ctk.IntVar(value=KNN_HP_OPTIM_DATA['p']['default_from']),
                to_var=ctk.IntVar(value=KNN_HP_OPTIM_DATA['p']['default_to']),
                min_val=KNN_HP_OPTIM_DATA['p']['min_val'],
                max_val=KNN_HP_OPTIM_DATA['p']['max_val'],
            ),
            'grid': {'row':0,'col':1,'colspan':1}
        },
        'weights': {
            'type': MultiSelectEntryField(
                options=KNN_HP_OPTIM_DATA['weights']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(KNN_HP_OPTIM_DATA['weights']['options']))
            ),
            'grid': {'row':1,'col':1,'colspan':1}
        },
        'algorithm': {
            'type': MultiSelectEntryField(
                options=KNN_HP_OPTIM_DATA['algorithm']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(KNN_HP_OPTIM_DATA['algorithm']['options']))
            ),
            'grid': {'row':2,'col':0,'colspan':2}
        },
    }
}
KNN_hp_optim_panel = HyperParamOptim_AlgoLabelFrame(
    master_panel=hyperparam_optim_panel,
    algoMap=ALGO_MAP,
    algo_name='KNN',
    algo_hp_optim_fields_data=KNN_HP_OPTIM_DATA_AND_VARS,
    my_font=MY_FONT_1,
    fg_color=COLORS['SKYBLUE_FG'],
    result_Loading_ImgPath=HP_OPTIMIZATION_ONGOING_IMG_PATH,
    SELECTED_FEATURES=SELECTED_FEATURES,
    trainVar=TRAIN_FILE_PATH,
    testVar=TEST_FILE_PATH,
    hyperParamsFrame_NumOfCells=2
)._GET_ALGO_LABELFRAME()

#GB
GB_HP_OPTIM_DATA = my_config_manager.get('algorithm_properties.GB.hp_optim')
GB_HP_OPTIM_DATA_AND_VARS = {
    'method_opts': GB_HP_OPTIM_DATA['method']['opts'],
    'method_selected_optVar': ctk.StringVar(value=GB_HP_OPTIM_DATA['method']['opts'][0]),
    'scoring_opts': GB_HP_OPTIM_DATA['scoring']['opts'],
    'scoring_selected_optVar': ctk.StringVar(value=GB_HP_OPTIM_DATA['scoring']['opts'][0]),
    'cvfolds_entryVar': ctk.IntVar(value=GB_HP_OPTIM_DATA['cvfolds']['default']),
    'cvfolds_min': GB_HP_OPTIM_DATA['cvfolds']['min'],
    'cvfolds_max': GB_HP_OPTIM_DATA['cvfolds']['max'],

    'hp_optim_field_configs': {
        'n_estimators': {
            'type': MyStepRangeEntryField(
                from_var=ctk.IntVar(value=GB_HP_OPTIM_DATA['n_estimators']['default_from']),
                to_var=ctk.IntVar(value=GB_HP_OPTIM_DATA['n_estimators']['default_to']),
                step_var=ctk.IntVar(value=GB_HP_OPTIM_DATA['n_estimators']['default_max_steps']),
                min_val=GB_HP_OPTIM_DATA['n_estimators']['min_val'],
                max_val=GB_HP_OPTIM_DATA['n_estimators']['max_val'],
                max_steps=GB_HP_OPTIM_DATA['n_estimators']['max_steps']
            ),
            'grid': {'row':0,'col':0,'colspan':1}
        },
        'learning_rate': {
            'type': MultiSelectEntryField(
                options=GB_HP_OPTIM_DATA['learning_rate']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(list(map(str, GB_HP_OPTIM_DATA['learning_rate']['options']))))
            ),
            'grid': {'row':1,'col':0,'colspan':1}
        },
        'criterion': {
            'type': MultiSelectEntryField(
                options=GB_HP_OPTIM_DATA['criterion']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(GB_HP_OPTIM_DATA['criterion']['options']))
            ),
            'grid': {'row':0,'col':1,'colspan':1}
        },
        'max_depth': {
            'type': MyRangeEntryField(
                from_var=ctk.IntVar(value=GB_HP_OPTIM_DATA['max_depth']['default_from']),
                to_var=ctk.IntVar(value=GB_HP_OPTIM_DATA['max_depth']['default_to']),
                min_val=GB_HP_OPTIM_DATA['max_depth']['min_val'],
                max_val=GB_HP_OPTIM_DATA['max_depth']['max_val'],
            ),
            'grid': {'row':1,'col':1,'colspan':1}
        },
        'min_samples_split': {
            'type': MyRangeEntryField(
                from_var=ctk.IntVar(value=GB_HP_OPTIM_DATA['min_samples_split']['default_from']),
                to_var=ctk.IntVar(value=GB_HP_OPTIM_DATA['min_samples_split']['default_to']),
                min_val=GB_HP_OPTIM_DATA['min_samples_split']['min_val'],
                max_val=GB_HP_OPTIM_DATA['min_samples_split']['max_val'],
            ),
            'grid': {'row':2,'col':0,'colspan':1}
        },
        'min_samples_leaf': {
            'type': MyRangeEntryField(
                from_var=ctk.IntVar(value=GB_HP_OPTIM_DATA['min_samples_leaf']['default_from']),
                to_var=ctk.IntVar(value=GB_HP_OPTIM_DATA['min_samples_leaf']['default_to']),
                min_val=GB_HP_OPTIM_DATA['min_samples_leaf']['min_val'],
                max_val=GB_HP_OPTIM_DATA['min_samples_leaf']['max_val'],
            ),
            'grid': {'row':2,'col':1,'colspan':1}
        }
    }
}
GB_hp_optim_panel = HyperParamOptim_AlgoLabelFrame(
    master_panel=hyperparam_optim_panel,
    algoMap=ALGO_MAP,
    algo_name='GB',
    algo_hp_optim_fields_data=GB_HP_OPTIM_DATA_AND_VARS,
    my_font=MY_FONT_1,
    fg_color=COLORS['SKYBLUE_FG'],
    result_Loading_ImgPath=HP_OPTIMIZATION_ONGOING_IMG_PATH,
    SELECTED_FEATURES=SELECTED_FEATURES,
    trainVar=TRAIN_FILE_PATH,
    testVar=TEST_FILE_PATH,
    hyperParamsFrame_NumOfCells=2
)._GET_ALGO_LABELFRAME()

MLP_HP_OPTIM_DATA = my_config_manager.get('algorithm_properties.MLP.hp_optim')
MLP_HP_OPTIM_DATA_AND_VARS = {
    'method_opts': MLP_HP_OPTIM_DATA['method']['opts'],
    'method_selected_optVar': ctk.StringVar(value=MLP_HP_OPTIM_DATA['method']['opts'][0]),
    'scoring_opts': MLP_HP_OPTIM_DATA['scoring']['opts'],
    'scoring_selected_optVar': ctk.StringVar(value=MLP_HP_OPTIM_DATA['scoring']['opts'][0]),
    'cvfolds_entryVar': ctk.IntVar(value=MLP_HP_OPTIM_DATA['cvfolds']['default']),
    'cvfolds_min': MLP_HP_OPTIM_DATA['cvfolds']['min'],
    'cvfolds_max': MLP_HP_OPTIM_DATA['cvfolds']['max'],

    'hp_optim_field_configs': {
        'hidden_layer_size': {
            'type': MyStepRangeEntryField(
                from_var=ctk.IntVar(value=MLP_HP_OPTIM_DATA['hidden_layer_size']['default_from']),
                to_var=ctk.IntVar(value=MLP_HP_OPTIM_DATA['hidden_layer_size']['default_to']),
                step_var=ctk.IntVar(value=MLP_HP_OPTIM_DATA['hidden_layer_size']['default_max_steps']),
                min_val=MLP_HP_OPTIM_DATA['hidden_layer_size']['min_val'],
                max_val=MLP_HP_OPTIM_DATA['hidden_layer_size']['max_val'],
                max_steps=MLP_HP_OPTIM_DATA['hidden_layer_size']['max_steps']
            ),
            'grid': {'row':0,'col':0,'colspan':1}
        },
        'num_of_hidden_layers': {
            'type': MyRangeEntryField(
                from_var=ctk.IntVar(value=MLP_HP_OPTIM_DATA['num_of_hidden_layers']['default_from']),
                to_var=ctk.IntVar(value=MLP_HP_OPTIM_DATA['num_of_hidden_layers']['default_to']),
                min_val=MLP_HP_OPTIM_DATA['num_of_hidden_layers']['min_val'],
                max_val=MLP_HP_OPTIM_DATA['num_of_hidden_layers']['max_val'],
            ),
            'grid': {'row':0,'col':1,'colspan':1}
        },
        'activation': {
            'type': MultiSelectEntryField(
                options=MLP_HP_OPTIM_DATA['activation']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(MLP_HP_OPTIM_DATA['activation']['options']))
            ),
            'grid': {'row':1,'col':0,'colspan':1}
        },
        'solver': {
            'type': MultiSelectEntryField(
                options=MLP_HP_OPTIM_DATA['solver']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(MLP_HP_OPTIM_DATA['solver']['options']))
            ),
            'grid': {'row':1,'col':1,'colspan':1}
        },
        'alpha': {
            'type': MultiSelectEntryField(
                options=MLP_HP_OPTIM_DATA['alpha']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(list(map(str, MLP_HP_OPTIM_DATA['alpha']['options']))))
            ),
            'grid': {'row':2,'col':0,'colspan':1}
        },
        'learning_rate': {
            'type': MultiSelectEntryField(
                options=MLP_HP_OPTIM_DATA['learning_rate']['options'],
                selected_opt_var=ctk.StringVar(value=','.join(MLP_HP_OPTIM_DATA['learning_rate']['options']))
            ),
            'grid': {'row':2,'col':1,'colspan':1}
        },
    }
}
MLP_hp_optim_panel = HyperParamOptim_AlgoLabelFrame(
    master_panel=hyperparam_optim_panel,
    algoMap=ALGO_MAP,
    algo_name='MLP',
    algo_hp_optim_fields_data=MLP_HP_OPTIM_DATA_AND_VARS,
    my_font=MY_FONT_1,
    fg_color=COLORS['SKYBLUE_FG'],
    result_Loading_ImgPath=HP_OPTIMIZATION_ONGOING_IMG_PATH,
    trainVar=TRAIN_FILE_PATH,
    testVar=TEST_FILE_PATH,
    SELECTED_FEATURES=SELECTED_FEATURES,

    hyperParamsFrame_NumOfCells=2
)._GET_ALGO_LABELFRAME()

# MODEL_BUILD ------------------------------------------------------------------------------------------
def CREATE_DATA_AND_VARS_MAP_FOR_MODEL_BUILD (DATA:dict, numOfFieldsPerRow: int = 4):
    DATA_AND_VARS = {'model_build_field_configs':{}}
    for fi, field in enumerate(DATA.keys()):
        if 'default_val' in DATA[field]:
            _type = StringField(
                variable_withDefaultVal=ctk.StringVar(value=str(DATA[field]['default_val']))
            )
        elif 'options' in DATA[field]:
            _type = OptionMenuField(
                variable_withSelectedOpt=ctk.StringVar(value=DATA[field]['options'][DATA[field]['default_idx']]),
                options=DATA[field]['options'],
            )
        DATA_AND_VARS['model_build_field_configs'][field] = {
            'type': _type,
            'grid': {'row':fi//numOfFieldsPerRow,'col':fi%numOfFieldsPerRow,'colspan':1}
        }
    return DATA_AND_VARS

model_build_panel = taskPanelMap['model_build']

model_build_featureAndAlgoSelectFrame = FeatureAndAlgorithmFrame(
    masterFrame=model_build_panel,
    my_font=MY_FONT_1,
    colors={
        'fg':COLORS['SKYBLUE_FG'], 
        'btn':{'fg':COLORS['GREY_FG'], 'hover':COLORS['GREY_FG']}
    },
    LOADED_FEATURES=ALL_LOADED_FEATURES,
    SELECTED_FEATURES=SELECTED_FEATURES,
    LIST_OF_ALGORITHMS=[algoVal['algo_name'] for algoVal in ALGO_MAP.values()],
    SELECTED_ALGORITHM=MODEL_BUILD_SELECTED_ALGORITHM,
    TRAIN_FILE_PATH=TRAIN_FILE_PATH,
    TEST_FILE_PATH=TEST_FILE_PATH
)

RF_MODEL_BUILD_DATA = my_config_manager.get('algorithm_properties.RF.model_build')
RF_MODEL_BUILD_DATA_AND_VARS = CREATE_DATA_AND_VARS_MAP_FOR_MODEL_BUILD(RF_MODEL_BUILD_DATA)

RF_model_build_panel = ModelBuild_AlgoLabelFrame(
    master_panel=model_build_panel,
    algoValueInMap=ALGO_MAP['RF'],
    algo_model_build_fields_data=RF_MODEL_BUILD_DATA_AND_VARS,
    my_font=MY_FONT_1,
    fg_color=COLORS['SKYBLUE_FG'],
    result_Loading_ImgPath=RESULTS_LOADING_IMG_PATH,
    SELECTED_FEATURES=SELECTED_FEATURES,
    trainVar=TRAIN_FILE_PATH,
    testVar=TEST_FILE_PATH,
    hyperParamsFrame_NumOfCells=4
)._GET_ALGO_LABELFRAME()

#SVM
SVM_MODEL_BUILD_DATA = my_config_manager.get('algorithm_properties.SVM.model_build')
SVM_MODEL_BUILD_DATA_AND_VARS = CREATE_DATA_AND_VARS_MAP_FOR_MODEL_BUILD(SVM_MODEL_BUILD_DATA)

SVM_model_build_panel = ModelBuild_AlgoLabelFrame(
    master_panel=model_build_panel,
    algoValueInMap=ALGO_MAP['SVM'],
    algo_model_build_fields_data=SVM_MODEL_BUILD_DATA_AND_VARS,
    my_font=MY_FONT_1,
    fg_color=COLORS['SKYBLUE_FG'],
    result_Loading_ImgPath=RESULTS_LOADING_IMG_PATH,
    SELECTED_FEATURES=SELECTED_FEATURES,
    trainVar=TRAIN_FILE_PATH,
    testVar=TEST_FILE_PATH,
    hyperParamsFrame_NumOfCells=4
)._GET_ALGO_LABELFRAME()

#LDA
LDA_MODEL_BUILD_DATA = my_config_manager.get('algorithm_properties.LDA.model_build')
LDA_MODEL_BUILD_DATA_AND_VARS = CREATE_DATA_AND_VARS_MAP_FOR_MODEL_BUILD(LDA_MODEL_BUILD_DATA, numOfFieldsPerRow=3)

LDA_model_build_panel = ModelBuild_AlgoLabelFrame(
    master_panel=model_build_panel,
    algoValueInMap=ALGO_MAP['LDA'],
    algo_model_build_fields_data=LDA_MODEL_BUILD_DATA_AND_VARS,
    my_font=MY_FONT_1,
    fg_color=COLORS['SKYBLUE_FG'],
    result_Loading_ImgPath=RESULTS_LOADING_IMG_PATH,
    SELECTED_FEATURES=SELECTED_FEATURES,
    trainVar=TRAIN_FILE_PATH,
    testVar=TEST_FILE_PATH,
    hyperParamsFrame_NumOfCells=3
)._GET_ALGO_LABELFRAME()

#LR
LR_MODEL_BUILD_DATA = my_config_manager.get('algorithm_properties.LR.model_build')
LR_MODEL_BUILD_DATA_AND_VARS = CREATE_DATA_AND_VARS_MAP_FOR_MODEL_BUILD(LR_MODEL_BUILD_DATA, numOfFieldsPerRow=4)

LR_model_build_panel = ModelBuild_AlgoLabelFrame(
    master_panel=model_build_panel,
    algoValueInMap=ALGO_MAP['LR'],
    algo_model_build_fields_data=LR_MODEL_BUILD_DATA_AND_VARS,
    my_font=MY_FONT_1,
    fg_color=COLORS['SKYBLUE_FG'],
    result_Loading_ImgPath=RESULTS_LOADING_IMG_PATH,
    SELECTED_FEATURES=SELECTED_FEATURES,
    trainVar=TRAIN_FILE_PATH,
    testVar=TEST_FILE_PATH,
    hyperParamsFrame_NumOfCells=4
)._GET_ALGO_LABELFRAME()

#KNN
KNN_MODEL_BUILD_DATA = my_config_manager.get('algorithm_properties.KNN.model_build')
KNN_MODEL_BUILD_DATA_AND_VARS = CREATE_DATA_AND_VARS_MAP_FOR_MODEL_BUILD(KNN_MODEL_BUILD_DATA, numOfFieldsPerRow=4)

KNN_model_build_panel = ModelBuild_AlgoLabelFrame(
    master_panel=model_build_panel,
    algoValueInMap=ALGO_MAP['KNN'],
    algo_model_build_fields_data=KNN_MODEL_BUILD_DATA_AND_VARS,
    my_font=MY_FONT_1,
    fg_color=COLORS['SKYBLUE_FG'],
    result_Loading_ImgPath=RESULTS_LOADING_IMG_PATH,
    SELECTED_FEATURES=SELECTED_FEATURES,
    trainVar=TRAIN_FILE_PATH,
    testVar=TEST_FILE_PATH,
    hyperParamsFrame_NumOfCells=4
)._GET_ALGO_LABELFRAME()


#GB
GB_MODEL_BUILD_DATA = my_config_manager.get('algorithm_properties.GB.model_build')
GB_MODEL_BUILD_DATA_AND_VARS = CREATE_DATA_AND_VARS_MAP_FOR_MODEL_BUILD(GB_MODEL_BUILD_DATA, numOfFieldsPerRow=4)

GB_model_build_panel = ModelBuild_AlgoLabelFrame(
    master_panel=model_build_panel,
    algoValueInMap=ALGO_MAP['GB'],
    algo_model_build_fields_data=GB_MODEL_BUILD_DATA_AND_VARS,
    my_font=MY_FONT_1,
    fg_color=COLORS['SKYBLUE_FG'],
    result_Loading_ImgPath=RESULTS_LOADING_IMG_PATH,
    SELECTED_FEATURES=SELECTED_FEATURES,
    trainVar=TRAIN_FILE_PATH,
    testVar=TEST_FILE_PATH,
    hyperParamsFrame_NumOfCells=4
)._GET_ALGO_LABELFRAME()

#MLP
MLP_MODEL_BUILD_DATA = my_config_manager.get('algorithm_properties.MLP.model_build')
MLP_MODEL_BUILD_DATA_AND_VARS = CREATE_DATA_AND_VARS_MAP_FOR_MODEL_BUILD(MLP_MODEL_BUILD_DATA, numOfFieldsPerRow=4)

MLP_model_build_panel = ModelBuild_AlgoLabelFrame(
    master_panel=model_build_panel,
    algoValueInMap=ALGO_MAP['MLP'],
    algo_model_build_fields_data=MLP_MODEL_BUILD_DATA_AND_VARS,
    my_font=MY_FONT_1,
    fg_color=COLORS['SKYBLUE_FG'],
    result_Loading_ImgPath=RESULTS_LOADING_IMG_PATH,
    SELECTED_FEATURES=SELECTED_FEATURES,
    trainVar=TRAIN_FILE_PATH,
    testVar=TEST_FILE_PATH,
    hyperParamsFrame_NumOfCells=4
)._GET_ALGO_LABELFRAME()

#----------------

ALGO_LB_FRAMES = {
    'hp_optim' : {
        'RF': RF_hp_optim_panel,
        'SVM': SVM_hp_optim_panel,
        'LDA': LDA_hp_optim_panel,
        'LR': LR_hp_optim_panel,
        'KNN': KNN_hp_optim_panel,
        'GB': GB_hp_optim_panel,
        'MLP': MLP_hp_optim_panel,
    },
    'model_build': {
        'RF': RF_model_build_panel,
        'SVM': SVM_model_build_panel,
        'LDA': LDA_model_build_panel,
        'LR': LR_model_build_panel,
        'KNN': KNN_model_build_panel,
        'GB': GB_model_build_panel,
        'MLP': MLP_model_build_panel
    }
}
ALGO_NAME_TO_KEY_MAP = { v['algo_name']:k for (k,v) in ALGO_MAP.items() }

# Configure hp_optim_algo_optionmenu to change to appropriate 'hp_optim_panel' on update
hp_optim_algo_optionmenu = hp_optim_featureAndAlgoSelectFrame._get_algorithm_optionmenu()
hp_optim_algo_optionmenu.configure(
    command = lambda selected_opt: showAlgoLabelFrame_ForOption_InTaskPanel(ALGO_LB_FRAMES['hp_optim'], ALGO_NAME_TO_KEY_MAP[selected_opt])
)
# Configure same for 'model_build_panel'
model_build_algo_optionmenu = model_build_featureAndAlgoSelectFrame._get_algorithm_optionmenu()
model_build_algo_optionmenu.configure(
    command = lambda selected_opt: showAlgoLabelFrame_ForOption_InTaskPanel(ALGO_LB_FRAMES['model_build'], ALGO_NAME_TO_KEY_MAP[selected_opt])
)


def showAlgoLabelFrame_ForOption_InTaskPanel(ALGO_LB_FRAMES: dict[str, ctk.CTkFrame], ALGO_OPT: str):
    ''' algoFrames: {RF: RF_lbFrame, SVM: SVM_lbFrame,..}, ALGO_OPT: [RF/SVM/LR/LDA/..] '''
    for algoFrame in ALGO_LB_FRAMES.values():
        algoFrame.grid_remove()
    ALGO_LB_FRAMES[ALGO_OPT].grid(row=1,column=0,rowspan=6, columnspan=7,sticky=ctk.NSEW,padx=8,pady=(2,5))
    ALGO_LB_FRAMES[ALGO_OPT].grid_columnconfigure(tuple(range(7)), weight=1)
 
# Setting Default hp_optim_selected_algorithm to index 0 and also displaying that frame (as set doesn't trigger command)
HP_OPTIM_SELECTED_ALGORITHM.set(ALGO_MAP['RF']['algo_name'])
showAlgoLabelFrame_ForOption_InTaskPanel(ALGO_LB_FRAMES['hp_optim'], 'RF')

MODEL_BUILD_SELECTED_ALGORITHM.set(ALGO_MAP['RF']['algo_name'])
showAlgoLabelFrame_ForOption_InTaskPanel(ALGO_LB_FRAMES['model_build'], 'RF')

# Settings Panel [Created only when 1st time we need settings... time-taking]
settings_panel = taskPanelMap['settings']
settingsFrame = None

settings_created_event = Event()  # Event to track when SettingsFrame is ready

def generate_settings_frame():
    global settingsFrame
    inProgress = InProgressWindow(parent=root, font=MY_FONT_1, gif_path=RESULTS_LOADING_IMG_PATH)
    inProgress.create()
    inProgress.update_progress_verdict('Creating Settings Panel..')
    
    def create_frame():
        global settingsFrame
        settingsFrame = SettingsFrame(
            masterFrame=settings_panel,
            inProgress=inProgress,
            my_font=MY_FONT_1,
            colors={
                'fg': COLORS['SKYBLUE_FG'], 
                'save_btn': {'fg': COLORS['MEDIUMGREEN_FG'], 'hover': COLORS['MEDIUMGREEN_HOVER_FG']},
                'reset_btn': {'fg': COLORS['GREY_FG'], 'hover': COLORS['GREY_HOVER_FG']},
                'optionmenu': {'fg': COLORS['GREY_FG'], 'hover': COLORS['GREY_HOVER_FG']},
            }
        )
        inProgress.destroy()  # Ensure inProgress is destroyed after frame creation
        settings_created_event.set()  # Signal that SettingsFrame is ready

    # Ensure GUI update happens in the main thread
    root.after(0, create_frame)

def create_settings_panel():
    global settingsFrame
    if settingsFrame:
        settings_created_event.set()  # If already created, signal readiness
        return
    
    print("Creating settings panel...")
    settings_created_event.clear()  # Reset event before creating
    
    # Run the settings frame creation in a separate thread
    thread = Thread(target=generate_settings_frame)
    thread.start()
    return thread  # Return the thread so we can wait for it

def create_settings_widgets_then_load_panel(func):
    """Ensure func() runs only after settings panel is fully created."""

    def wrapper():
        thread = create_settings_panel()
        if thread:
            print('Waiting for settings panel to be created...')
            settings_created_event.wait()  # Wait until settingsFrame is ready
            print('Settings panel created, running func...')
        func()  # Now execute the provided function
    
    # Run wrapper in a separate thread to avoid blocking the main UI thread
    Thread(target=wrapper).start()

# Ensure clicking the button waits for settings to load before executing the next command
cmd_to_load_settings_panel = taskSelectBtns['settings'].cget('command')
taskSelectBtns['settings'].configure(command=lambda *args: create_settings_widgets_then_load_panel(cmd_to_load_settings_panel))

root.mainloop()