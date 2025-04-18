from customtkinter import *
from util.gui.widgets import InProgressWindow, CustomWarningBox, CustomSuccessBox
from threading import Thread
from json import dumps as jsonDumps
from util.ml.functions import *
from ast import literal_eval as ast_literal_eval

def convertStrToIntOrFloat(value: str):
    return float(value) if '.' in value else int(value)
    
def format_list_display(A:list):
    nE = len(A)
    if nE>6:
        return f'[{str(A[:3])[1:-1]},..,{str(A[-2:])[1:-1]}]'
    else:
        return str(A)

def RF_HP_OPTIM_SUBMIT (master:CTk, loading_gif_path:str, RF_inputs: dict, RF_resultsVar: StringVar, font: CTkFont, trainEntryVar: StringVar, testEntryVar: StringVar):
    inProgress = InProgressWindow(master, font, loading_gif_path)
    inProgress.create()

    def update_success (processOutput: dict):
        inProgress.destroy()
        RF_resultsVar.set(jsonDumps(processOutput, indent=4))
        CustomSuccessBox(master, "Calculations Completed !!", font)
        
    def update_failure (warnings: list):
        inProgress.destroy()
        RF_resultsVar.set('..')
        CustomWarningBox(master, warnings, font)
    
    # FEATURES
    if not len(RF_inputs["FEATURES"].get()):
        master.after(1000, lambda warnings=['No FEATURES selected !!']: update_failure(warnings))
        return

    RF_inputs = {
        'FEATURES': RF_inputs["FEATURES"].get().split(','),
        'METHOD': RF_inputs['METHOD'].get(),
        'SCORING': RF_inputs['SCORING'].get(),
        'CROSS_FOLD_VALID': RF_inputs['CROSS_FOLD_VALID'].get(),
        'n_estimators': {k:v.get() for k,v in RF_inputs['n_estimators'].items()},
        'criterion': RF_inputs['criterion'].get().split(','),
        'max_depth': {k:v.get() for k,v in RF_inputs['max_depth'].items()},
        'min_samples_split': {k:v.get() for k,v in RF_inputs['min_samples_split'].items()},
        'min_samples_leaf': {k:v.get() for k,v in RF_inputs['min_samples_leaf'].items()},
    }

    # GUI remains responsive on main thread, the optimization runs on seperate thread
    def RUN_OPTIMIZATION():
        try:
            RF_resultsVar.set('...')
            processResult = RF_HP_OPTIM_PROCESS (
                HP_OPTIM_INPUTS=RF_inputs, IN_PROGRESS = inProgress,
                TRAIN_FILE_PATH=trainEntryVar.get(), TEST_FILE_PATH=testEntryVar.get()
            )
            master.after(1000, lambda processOut=processResult: update_success(processOut))
        except Exception as ex:
            master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))
    
    Thread(target=RUN_OPTIMIZATION).start()

def REPLOT_OPTUNA_SUBMIT (master:CTk, loading_gif_path:str, font: CTkFont, algo_name:str, study_name:str):
    inProgress = InProgressWindow(master, font, loading_gif_path)
    inProgress.create()
    inProgress.update_progress_verdict('RE-Generating Plots')

    def update_success():
        inProgress.destroy()
        CustomSuccessBox(master, "All Plots Are Regenerated Successfully !!", font)
        
    def update_failure (warnings: list):
        inProgress.destroy()
        CustomWarningBox(master, warnings, font)
    
    # GUI remains responsive on main thread, the optimization runs on seperate thread
    def REPLOT():
        try:
            from util.ml.functions import RE_PLOT_OPTUNA_GRAPHS
            RE_PLOT_OPTUNA_GRAPHS(algo_name, study_name, inProgress)
            master.after(1000, update_success)
        except Exception as ex:
            master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))
    
    Thread(target=REPLOT).start()

def SVM_HP_OPTIM_SUBMIT (master:CTk, loading_gif_path:str, SVM_inputs: dict, SVM_resultsVar: StringVar, font: CTkFont, trainEntryVar: StringVar, testEntryVar: StringVar):
    inProgress = InProgressWindow(master, font, loading_gif_path)
    inProgress.create()

    def update_success (processOutput: dict):
        inProgress.destroy()
        SVM_resultsVar.set(jsonDumps(processOutput, indent=4))
        CustomSuccessBox(master, "Calculations Completed !!", font)
        
    def update_failure (warnings: list):
        inProgress.destroy()
        SVM_resultsVar.set('..')
        CustomWarningBox(master, warnings, font)
    
    # FEATURES
    if not len(SVM_inputs["FEATURES"].get()):
        master.after(1000, lambda warnings=['No FEATURES selected !!']: update_failure(warnings))
        return

    SVM_inputs = {
        'FEATURES': SVM_inputs["FEATURES"].get().split(','),
        'METHOD': SVM_inputs['METHOD'].get(),
        'SCORING': SVM_inputs['SCORING'].get(),
        'CROSS_FOLD_VALID': SVM_inputs['CROSS_FOLD_VALID'].get(),
        'C': list(map(float, SVM_inputs['C'].get().split(','))),
        'gamma': SVM_inputs['gamma'].get().split(','),
        'kernel': SVM_inputs['kernel'].get().split(','),
    }

    # GUI remains responsive on main thread, the optimization runs on seperate thread
    def RUN_OPTIMIZATION():
        try:
            SVM_resultsVar.set('...')
            processResult = SVM_HP_OPTIM_PROCESS (
                HP_OPTIM_INPUTS=SVM_inputs, IN_PROGRESS = inProgress,
                TRAIN_FILE_PATH=trainEntryVar.get(), TEST_FILE_PATH=testEntryVar.get()
            )
            master.after(1000, lambda processOut=processResult: update_success(processOut))
        except Exception as ex:
            master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))
    
    Thread(target=RUN_OPTIMIZATION).start()

def LR_HP_OPTIM_SUBMIT (master:CTk, loading_gif_path:str, LR_inputs: dict, LR_resultsVar: StringVar, font: CTkFont, trainEntryVar: StringVar, testEntryVar: StringVar):
    inProgress = InProgressWindow(master, font, loading_gif_path)
    inProgress.create()

    def update_success (processOutput: dict):
        inProgress.destroy()
        LR_resultsVar.set(jsonDumps(processOutput, indent=4))
        CustomSuccessBox(master, "Calculations Completed !!", font)
        
    def update_failure (warnings: list):
        inProgress.destroy()
        LR_resultsVar.set('..')
        CustomWarningBox(master, warnings, font)
    
    # FEATURES
    if not len(LR_inputs["FEATURES"].get()):
        master.after(1000, lambda warnings=['No FEATURES selected !!']: update_failure(warnings))
        return

    LR_inputs = {
        'FEATURES': LR_inputs["FEATURES"].get().split(','),
        'METHOD': LR_inputs['METHOD'].get(),
        'SCORING': LR_inputs['SCORING'].get(),
        'CROSS_FOLD_VALID': LR_inputs['CROSS_FOLD_VALID'].get(),
        'C': list(map(float, LR_inputs['C'].get().split(','))),
        'penalty': LR_inputs['penalty'].get().split(','),
        'solver': LR_inputs['solver'].get().split(','),
    }

    # GUI remains responsive on main thread, the optimization runs on seperate thread
    def RUN_OPTIMIZATION():
        try:
            LR_resultsVar.set('...')
            processResult = LR_HP_OPTIM_PROCESS (
                HP_OPTIM_INPUTS=LR_inputs, IN_PROGRESS = inProgress,
                TRAIN_FILE_PATH=trainEntryVar.get(), TEST_FILE_PATH=testEntryVar.get()
            )
            master.after(1000, lambda processOut=processResult: update_success(processOut))
        except Exception as ex:
            master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))
    
    Thread(target=RUN_OPTIMIZATION).start()

def LDA_HP_OPTIM_SUBMIT (master:CTk, loading_gif_path:str, LDA_inputs: dict, LDA_resultsVar: StringVar, font: CTkFont, trainEntryVar: StringVar, testEntryVar: StringVar):
    inProgress = InProgressWindow(master, font, loading_gif_path)
    inProgress.create()

    def update_success (processOutput: dict):
        inProgress.destroy()
        LDA_resultsVar.set(jsonDumps(processOutput, indent=4))
        CustomSuccessBox(master, "Calculations Completed !!", font)
        
    def update_failure (warnings: list):
        inProgress.destroy()
        LDA_resultsVar.set('..')
        CustomWarningBox(master, warnings, font)
    
    # FEATURES
    if not len(LDA_inputs["FEATURES"].get()):
        master.after(1000, lambda warnings=['No FEATURES selected !!']: update_failure(warnings))
        return

    LDA_inputs = {
        'FEATURES': LDA_inputs["FEATURES"].get().split(','),
        'METHOD': LDA_inputs['METHOD'].get(),
        'SCORING': LDA_inputs['SCORING'].get(),
        'CROSS_FOLD_VALID': LDA_inputs['CROSS_FOLD_VALID'].get(),
        'solver': LDA_inputs['solver'].get().split(','),
    }

    # GUI remains responsive on main thread, the optimization runs on seperate thread
    def RUN_OPTIMIZATION():
        try:
            LDA_resultsVar.set('...')
            processResult = LDA_HP_OPTIM_PROCESS (
                HP_OPTIM_INPUTS=LDA_inputs, IN_PROGRESS = inProgress,
                TRAIN_FILE_PATH=trainEntryVar.get(), TEST_FILE_PATH=testEntryVar.get()
            )
            master.after(1000, lambda processOut=processResult: update_success(processOut))
        except Exception as ex:
            master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))
    
    Thread(target=RUN_OPTIMIZATION).start()

def KNN_HP_OPTIM_SUBMIT (master:CTk, loading_gif_path:str, KNN_inputs: dict, KNN_resultsVar: StringVar, font: CTkFont, trainEntryVar: StringVar, testEntryVar: StringVar):
    inProgress = InProgressWindow(master, font, loading_gif_path)
    inProgress.create()

    def update_success (processOutput: dict):
        inProgress.destroy()
        KNN_resultsVar.set(jsonDumps(processOutput, indent=4))
        CustomSuccessBox(master, "Calculations Completed !!", font)
        
    def update_failure (warnings: list):
        inProgress.destroy()
        KNN_resultsVar.set('..')
        CustomWarningBox(master, warnings, font)
    
    # FEATURES
    if not len(KNN_inputs["FEATURES"].get()):
        master.after(1000, lambda warnings=['No FEATURES selected !!']: update_failure(warnings))
        return

    KNN_inputs = {
        'FEATURES': KNN_inputs["FEATURES"].get().split(','),
        'METHOD': KNN_inputs['METHOD'].get(),
        'SCORING': KNN_inputs['SCORING'].get(),
        'CROSS_FOLD_VALID': KNN_inputs['CROSS_FOLD_VALID'].get(),
        'p': {k:v.get() for k,v in KNN_inputs['p'].items()},
        'leaf_size': {k:v.get() for k,v in KNN_inputs['leaf_size'].items()},
        'n_neighbors': {k:v.get() for k,v in KNN_inputs['n_neighbors'].items()},
        'weights': KNN_inputs['weights'].get().split(','),
        'algorithm': KNN_inputs['algorithm'].get().split(','),
    }

    # GUI remains responsive on main thread, the optimization runs on seperate thread
    def RUN_OPTIMIZATION():
        try:
            KNN_resultsVar.set('...')
            processResult = KNN_HP_OPTIM_PROCESS (
                HP_OPTIM_INPUTS=KNN_inputs, IN_PROGRESS = inProgress,
                TRAIN_FILE_PATH=trainEntryVar.get(), TEST_FILE_PATH=testEntryVar.get()
            )
            master.after(1000, lambda processOut=processResult: update_success(processOut))
        except Exception as ex:
            master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))
    
    Thread(target=RUN_OPTIMIZATION).start()

def GB_HP_OPTIM_SUBMIT (master:CTk, loading_gif_path:str, GB_inputs: dict, GB_resultsVar: StringVar, font: CTkFont, trainEntryVar: StringVar, testEntryVar: StringVar):
    inProgress = InProgressWindow(master, font, loading_gif_path)
    inProgress.create()

    def update_success (processOutput: dict):
        inProgress.destroy()
        GB_resultsVar.set(jsonDumps(processOutput, indent=4))
        CustomSuccessBox(master, "Calculations Completed !!", font)
        
    def update_failure (warnings: list):
        inProgress.destroy()
        GB_resultsVar.set('..')
        CustomWarningBox(master, warnings, font)
    
    # FEATURES
    if not len(GB_inputs["FEATURES"].get()):
        master.after(1000, lambda warnings=['No FEATURES selected !!']: update_failure(warnings))
        return

    GB_inputs = {
        'FEATURES': GB_inputs["FEATURES"].get().split(','),
        'METHOD': GB_inputs['METHOD'].get(),
        'SCORING': GB_inputs['SCORING'].get(),
        'CROSS_FOLD_VALID': GB_inputs['CROSS_FOLD_VALID'].get(),
        'n_estimators': {k:v.get() for k,v in GB_inputs['n_estimators'].items()},
        'learning_rate': list(map(float, GB_inputs['learning_rate'].get().split(','))),
        'criterion': GB_inputs['criterion'].get().split(','),
        'max_depth': {k:v.get() for k,v in GB_inputs['max_depth'].items()},
        'min_samples_split': {k:v.get() for k,v in GB_inputs['min_samples_split'].items()},
        'min_samples_leaf': {k:v.get() for k,v in GB_inputs['min_samples_leaf'].items()},
    }

    # GUI remains responsive on main thread, the optimization runs on seperate thread
    def RUN_OPTIMIZATION():
        try:
            GB_resultsVar.set('...')
            processResult = GB_HP_OPTIM_PROCESS (
                HP_OPTIM_INPUTS=GB_inputs, IN_PROGRESS = inProgress,
                TRAIN_FILE_PATH=trainEntryVar.get(), TEST_FILE_PATH=testEntryVar.get()
            )
            master.after(1000, lambda processOut=processResult: update_success(processOut))
        except Exception as ex:
            master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))
    
    Thread(target=RUN_OPTIMIZATION).start()

def MLP_HP_OPTIM_SUBMIT (master:CTk, loading_gif_path:str, MLP_inputs: dict, MLP_resultsVar: StringVar, font: CTkFont, trainEntryVar: StringVar, testEntryVar: StringVar):
    inProgress = InProgressWindow(master, font, loading_gif_path)
    inProgress.create()

    def update_success (processOutput: dict):
        inProgress.destroy()
        MLP_resultsVar.set(jsonDumps(processOutput, indent=4))
        CustomSuccessBox(master, "Calculations Completed !!", font)
        
    def update_failure (warnings: list):
        inProgress.destroy()
        MLP_resultsVar.set('..')
        CustomWarningBox(master, warnings, font)
    
    # FEATURES
    if not len(MLP_inputs["FEATURES"].get()):
        master.after(1000, lambda warnings=['No FEATURES selected !!']: update_failure(warnings))
        return

    MLP_inputs = {
        'FEATURES': MLP_inputs["FEATURES"].get().split(','),
        'METHOD': MLP_inputs['METHOD'].get(),
        'SCORING': MLP_inputs['SCORING'].get(),
        'CROSS_FOLD_VALID': MLP_inputs['CROSS_FOLD_VALID'].get(),
        'hidden_layer_size': {k:v.get() for k,v in MLP_inputs['hidden_layer_size'].items()},
        'num_of_hidden_layers': {k:v.get() for k,v in MLP_inputs['num_of_hidden_layers'].items()},
        'activation': MLP_inputs['activation'].get().split(','),
        'solver': MLP_inputs['solver'].get().split(','),
        'alpha': MLP_inputs['alpha'].get().split(','),
        'learning_rate': MLP_inputs['learning_rate'].get().split(','),
    }

    # GUI remains responsive on main thread, the optimization runs on seperate thread
    def RUN_OPTIMIZATION():
        MLP_resultsVar.set('...')
        processResult = MLP_HP_OPTIM_PROCESS (
            HP_OPTIM_INPUTS=MLP_inputs, IN_PROGRESS = inProgress,
            TRAIN_FILE_PATH=trainEntryVar.get(), TEST_FILE_PATH=testEntryVar.get()
        )
        master.after(1000, lambda processOut=processResult: update_success(processOut))
        # try:
        #     MLP_resultsVar.set('...')
        #     processResult = MLP_HP_OPTIM_PROCESS (
        #         HP_OPTIM_INPUTS=MLP_inputs, IN_PROGRESS = inProgress,
        #         TRAIN_FILE_PATH=trainEntryVar.get(), TEST_FILE_PATH=testEntryVar.get()
        #     )
        #     master.after(1000, lambda processOut=processResult: update_success(processOut))
        # except Exception as ex:
        #     master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))
    
    Thread(target=RUN_OPTIMIZATION).start()


def RF_MODEL_BUILD_SUBMIT (master:CTk, loading_gif_path:str, RFmb_inputs: dict, RFmb_resultsVar: StringVar, font: CTkFont, trainEntryVar: StringVar, testEntryVar: StringVar):
    inProgress = InProgressWindow(master, font, loading_gif_path)
    inProgress.create()

    def update_success (processOutput: dict):
        inProgress.destroy()
        RFmb_resultsVar.set(jsonDumps(processOutput, indent=4))
        CustomSuccessBox(master, "Calculations Completed !!", font)
        
    def update_failure (warnings: list):
        inProgress.destroy()
        RFmb_resultsVar.set('..')
        CustomWarningBox(master, warnings, font)
    
    RFmb_out = {k:v.get() for k,v in RFmb_inputs.items()}
    WARNINGS = []

    # FEATURES
    if not len(RFmb_out["FEATURES"]):
        master.after(1000, lambda warnings=['No FEATURES selected !!']: update_failure(warnings))
        return
    else:
        RFmb_out["FEATURES"] = RFmb_out["FEATURES"].split(',')

    # n_estimators
    try:
        RFmb_out["n_estimators"] = int(RFmb_out["n_estimators"])
    except:
        WARNINGS.append("n_estimators must be an INTEGER")

    # max_depth
    try:
        RFmb_out["max_depth"] = None if RFmb_out["max_depth"].lower()=='none' else int(RFmb_out["max_depth"])
    except:
        WARNINGS.append("max_depth must be an INTEGER or 'None'")

    # min_samples_split
    try:
        RFmb_out["min_samples_split"] = convertStrToIntOrFloat(RFmb_out["min_samples_split"])
    except:
        WARNINGS.append('min_samples_split must be int or float')
    
    # min_samples_leaf
    try:
        RFmb_out["min_samples_leaf"] = convertStrToIntOrFloat(RFmb_out["min_samples_leaf"])
    except:
        WARNINGS.append('min_samples_leaf must be int or float')

    # min_impurity_decrease
    try:
        RFmb_out["min_impurity_decrease"] = float(RFmb_out["min_impurity_decrease"])
    except:
        WARNINGS.append('min_impurity_decrease must be float')

    # random_state
    try:
        RFmb_out["random_state"] = None if RFmb_out["random_state"].lower()=='none' else int(RFmb_out["random_state"])
    except:
        WARNINGS.append("random_state must be an INTEGER or 'None'")

    # WARM START
    RFmb_out["warm_start"] = True if RFmb_out["warm_start"].lower()=='true' else False

    # max_features
    try:
        if RFmb_out["max_features"].lower()=='none':
            RFmb_out["max_features"] = None
        elif RFmb_out["max_features"].lower() in ['sqrt', 'log2']:
            RFmb_out["max_features"] = RFmb_out["max_features"].lower()
        else:
            RFmb_out["max_features"] = convertStrToIntOrFloat(RFmb_out["max_features"])
    except:
        WARNINGS.append("max_features must be an INTEGER, FLOAT or either of 'sqrt'/'log2'/None")
    
    # min_weight_fraction_leaf
    try:
        RFmb_out["min_weight_fraction_leaf"] = float(RFmb_out["min_weight_fraction_leaf"])
    except:
        WARNINGS.append('min_weight_fraction_leaf must be float')
    
    # max_leaf_nodes
    try:
        RFmb_out["max_leaf_nodes"] = None if RFmb_out["max_leaf_nodes"].lower()=='none' else int(RFmb_out["max_leaf_nodes"])
    except:
        WARNINGS.append("max_leaf_nodes must be an INTEGER")
    
    def func_to_update_inProgress_text(txt):
        inProgress.update_progress_verdict(txt)

    def run_process():
        try:
            processResultDict = RF_MODEL_BUILD_PROCESS(
                RfMB_ValidatedInputs=RFmb_out, 
                trainFilePath=trainEntryVar.get(),
                testFilePath=testEntryVar.get(),
                inProgressUpdateFunc=func_to_update_inProgress_text
            )
            master.after(1000, lambda processOut=processResultDict: update_success(processOut))
        except Exception as ex:
            master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))

    if len(WARNINGS) == 0:
        print("[RF] VALIDATED_INPUTS: ", jsonDumps(RFmb_out, indent=2))
        Thread(target=run_process).start()
    else:
        master.after(1000, lambda warnings=WARNINGS: update_failure(warnings))

def SVM_MODEL_BUILD_SUBMIT (master:CTk, loading_gif_path:str, SVMmb_inputs: dict, SVMmb_resultsVar: StringVar, font: CTkFont, trainEntryVar: StringVar, testEntryVar: StringVar):
    inProgress = InProgressWindow(master, font, loading_gif_path)
    inProgress.create()
    
    def update_success (processOutput: dict):
        inProgress.destroy()
        SVMmb_resultsVar.set(jsonDumps(processOutput, indent=4))
        CustomSuccessBox(master, "Calculations Completed !!", font)
        
    def update_failure (warnings: list):
        inProgress.destroy()
        SVMmb_resultsVar.set('..')
        CustomWarningBox(master, warnings, font)

    SVMmb_out = {k:v.get() for k,v in SVMmb_inputs.items()}
    WARNINGS = []

    # FEATURES
    if not len(SVMmb_out["FEATURES"]):
        master.after(1000, lambda warnings=['No FEATURES selected !!']: update_failure(warnings))
        return
    else:
        SVMmb_out["FEATURES"] = SVMmb_out["FEATURES"].split(',')

    # C
    try:
        SVMmb_out["C"] = float(SVMmb_out["C"])
    except:
        WARNINGS.append('C must be int or float')

    # degree
    try:
        SVMmb_out["degree"] = int(SVMmb_out["degree"])
    except:
        WARNINGS.append("degree must be an INTEGER")

    # coef0
    try:
        SVMmb_out["coef0"] = float(SVMmb_out["coef0"])
    except:
        WARNINGS.append('coef0 must be float')

    # tol
    try:
        SVMmb_out["tol"] = float(SVMmb_out["tol"])
    except:
        WARNINGS.append('tol must be float')

    # shrinking
    SVMmb_out["shrinking"] = True if SVMmb_out["shrinking"].lower()=='true' else False
    # probability
    SVMmb_out["probability"] = True if SVMmb_out["probability"].lower()=='true' else False
    # break_ties
    SVMmb_out["break_ties"] = True if SVMmb_out["break_ties"].lower()=='true' else False

    # random_state
    try:
        SVMmb_out["random_state"] = None if SVMmb_out["random_state"].lower()=='none' else int(SVMmb_out["random_state"])
    except:
        WARNINGS.append("random_state must be an INTEGER or 'None'")
    
    def func_to_update_inProgress_text(txt):
        inProgress.update_progress_verdict(txt)

    def run_process ():
        try:
            processResultDict = SVM_MODEL_BUILD_PROCESS(
                SvmMB_ValidatedInputs=SVMmb_out, 
                trainFilePath=trainEntryVar.get(),
                testFilePath=testEntryVar.get(),
                inProgressUpdateFunc=func_to_update_inProgress_text
            )
            print("DONE PROCESS")
            master.after(1000, lambda processOut=processResultDict: update_success(processOut))
        except Exception as ex:
            master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))

    if len(WARNINGS) == 0:
        print("[SVM] VALIDATED_INPUTS: ", jsonDumps(SVMmb_out, indent=2))
        # Run this Heavy Task on seperate Thread, 
        # Tkinter is singlethreaded with mainLoop() running on it, handles the GUI and TopLevel will show
        # Running the run_process on main Thread will hang the GUI, arrest the main thread
        Thread(target=run_process).start() 
    else:
        master.after(1000, lambda warnings=WARNINGS: update_failure(warnings))

def LDA_MODEL_BUILD_SUBMIT (master:CTk, loading_gif_path:str, LDAmb_inputs: dict, LDAmb_resultsVar: StringVar, font: CTkFont, trainEntryVar: StringVar, testEntryVar: StringVar):
    inProgress = InProgressWindow(master, font, loading_gif_path)
    inProgress.create()
    
    def update_success (processOutput: dict):
        inProgress.destroy()
        LDAmb_resultsVar.set(jsonDumps(processOutput, indent=4))
        CustomSuccessBox(master, "Calculations Completed !!", font)
        
    def update_failure (warnings: list):
        inProgress.destroy()
        LDAmb_resultsVar.set('..')
        CustomWarningBox(master, warnings, font)

    LDAmb_out = {k:v.get() for k,v in LDAmb_inputs.items()}
    WARNINGS = []

    # FEATURES
    if not len(LDAmb_out["FEATURES"]):
        master.after(1000, lambda warnings=['No FEATURES selected !!']: update_failure(warnings))
        return
    else:
        LDAmb_out["FEATURES"] = LDAmb_out["FEATURES"].split(',')

    # shrinkage
    try:
        if LDAmb_out["shrinkage"].lower()=='none':
            LDAmb_out["shrinkage"] = None
        elif LDAmb_out["shrinkage"].lower()=='auto':
            LDAmb_out["shrinkage"] = 'auto'
        else:
            LDAmb_out["shrinkage"] = float(LDAmb_out["shrinkage"])
    except:
        WARNINGS.append("shrinkage must be a float, 'auto' or None")

    # n_components
    try:
        LDAmb_out["n_components"] = None if LDAmb_out["n_components"].lower()=='none' else int(LDAmb_out["n_components"])
    except:
        WARNINGS.append("n_components must be an integer or None")

    # tol
    try:
        LDAmb_out["tol"] = float(LDAmb_out["tol"])
    except:
        WARNINGS.append('tol must be float')

    # store_covariance
    LDAmb_out["store_covariance"] = True if LDAmb_out["store_covariance"].lower()=='true' else False

    def func_to_update_inProgress_text(txt):
        inProgress.update_progress_verdict(txt)

    def run_process ():
        try:
            processResultDict = LDA_MODEL_BUILD_PROCESS(
                LdaMB_ValidatedInputs=LDAmb_out, 
                trainFilePath=trainEntryVar.get(),
                testFilePath=testEntryVar.get(),
                inProgressUpdateFunc=func_to_update_inProgress_text
            )
            master.after(1000, lambda processOut=processResultDict: update_success(processOut))
        except Exception as ex:
            master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))

    if len(WARNINGS) == 0:
        print("[LDA] VALIDATED_INPUTS: ", jsonDumps(LDAmb_out, indent=2))
        Thread(target=run_process).start()
    else:
        master.after(1000, lambda warnings=WARNINGS: update_failure(warnings))

def LR_MODEL_BUILD_SUBMIT (master:CTk, loading_gif_path:str, LRmb_inputs: dict, LRmb_resultsVar: StringVar, font: CTkFont, trainEntryVar: StringVar, testEntryVar: StringVar):
    inProgress = InProgressWindow(master, font, loading_gif_path)
    inProgress.create()
    
    def update_success (processOutput: dict):
        inProgress.destroy()
        LRmb_resultsVar.set(jsonDumps(processOutput, indent=4))
        CustomSuccessBox(master, "Calculations Completed !!", font)
        
    def update_failure (warnings: list):
        inProgress.destroy()
        LRmb_resultsVar.set('..')
        CustomWarningBox(master, warnings, font)

    LRmb_out = {k:v.get() for k,v in LRmb_inputs.items()}
    WARNINGS = []

    # FEATURES
    if not len(LRmb_out["FEATURES"]):
        master.after(1000, lambda warnings=['No FEATURES selected !!']: update_failure(warnings))
        return
    else:
        LRmb_out["FEATURES"] = LRmb_out["FEATURES"].split(',')

    # l1_ratio
    try:
        LRmb_out["l1_ratio"] = None if LRmb_out["l1_ratio"].lower()=='none' else float(LRmb_out["l1_ratio"])
    except:
        WARNINGS.append('l1_ratio must be float or None')

    # penalty
    if LRmb_out["penalty"].lower()=='none':
        LRmb_out["penalty"] = None
    
    # tol
    try:
        LRmb_out["tol"] = float(LRmb_out["tol"])
    except:
        WARNINGS.append('tol must be float')

    # C
    try:
        LRmb_out["C"] = float(LRmb_out["C"])
    except:
        WARNINGS.append('C must be float')

    # intercept_scaling
    try:
        LRmb_out["intercept_scaling"] = float(LRmb_out["intercept_scaling"])
    except:
        WARNINGS.append('intercept_scaling must be int or float')

    # random_state
    try:
        LRmb_out["random_state"] = None if LRmb_out["random_state"].lower()=='none' else int(LRmb_out["random_state"])
    except:
        WARNINGS.append("random_state must be an INTEGER or 'None'")

    # max_iter
    try:
        LRmb_out["max_iter"] = int(LRmb_out["max_iter"])
    except:
        WARNINGS.append('max_iter must be int')
    
    # n_jobs
    try:
        LRmb_out["n_jobs"] = None if LRmb_out["n_jobs"].lower()=='none' else int(LRmb_out["n_jobs"])
    except:
        WARNINGS.append("n_jobs must be an integer or None")
    
    # fit_intercept
    LRmb_out["fit_intercept"] = True if LRmb_out["fit_intercept"].lower()=='true' else False
    # warm_start
    LRmb_out["warm_start"] = True if LRmb_out["warm_start"].lower()=='true' else False

    def func_to_update_inProgress_text(txt):
        inProgress.update_progress_verdict(txt)

    def run_process ():
        try:
            processResultDict = LR_MODEL_BUILD_PROCESS(
                LrMB_ValidatedInputs=LRmb_out, 
                trainFilePath=trainEntryVar.get(),
                testFilePath=testEntryVar.get(),
                inProgressUpdateFunc=func_to_update_inProgress_text
            )
            master.after(1000, lambda processOut=processResultDict: update_success(processOut))
        except Exception as ex:
            master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))

    if len(WARNINGS) == 0:
        print("[LR] VALIDATED_INPUTS: ", jsonDumps(LRmb_out, indent=2))
        Thread(target=run_process).start()
    else:
        master.after(1000, lambda warnings=WARNINGS: update_failure(warnings))

def KNN_MODEL_BUILD_SUBMIT (master:CTk, loading_gif_path:str, KNNmb_inputs: dict, KNNmb_resultsVar: StringVar, font: CTkFont, trainEntryVar: StringVar, testEntryVar: StringVar):
    
    inProgress = InProgressWindow(master, font, loading_gif_path)
    inProgress.create()

    def update_success (processOutput: dict):
        inProgress.destroy()
        KNNmb_resultsVar.set(jsonDumps(processOutput, indent=4))
        CustomSuccessBox(master, "Calculations Completed !!", font)
        
    def update_failure (warnings: list):
        inProgress.destroy()
        KNNmb_resultsVar.set('..')
        CustomWarningBox(master, warnings, font)
    
    KNNmb_out = {k:v.get() for k,v in KNNmb_inputs.items()}
    WARNINGS = []

    # FEATURES
    if not len(KNNmb_out["FEATURES"]):
        master.after(1000, lambda warnings=['No FEATURES selected !!']: update_failure(warnings))
        return
    else:
        KNNmb_out["FEATURES"] = KNNmb_out["FEATURES"].split(',')

    # n_neighbors
    try:
        KNNmb_out["n_neighbors"] = int(KNNmb_out["n_neighbors"])
    except:
        WARNINGS.append("n_neighbors must be an INTEGER")

    # leaf_size
    try:
        KNNmb_out["leaf_size"] = int(KNNmb_out["leaf_size"])
    except:
        WARNINGS.append("leaf_size must be an INTEGER")

    # algorithm
    try:
        if KNNmb_out["algorithm"].lower() in ['auto','ball_tree','kd_tree','brute']:
            KNNmb_out["algorithm"] = KNNmb_out["algorithm"].lower()
        else:
            KNNmb_out["algorithm"] = convertStrToIntOrFloat(KNNmb_out["algorithm"])
    except:
        WARNINGS.append("algorithm must be either of 'auto'/'ball_tree'/'kd_tree'/'brute'")

    # p
    try:
        KNNmb_out["p"] = float(KNNmb_out["p"])
    except:
        WARNINGS.append('p must be float')

    # n_jobs
    try:
        KNNmb_out["n_jobs"] = None if KNNmb_out["n_jobs"].lower()=='none' else int(KNNmb_out["n_jobs"])
    except:
        WARNINGS.append("n_jobs must be an INTEGER or 'None'")

    def func_to_update_inProgress_text(txt):
        inProgress.update_progress_verdict(txt)

    def run_process ():
        try:
            processResultDict = KNN_MODEL_BUILD_PROCESS(
                KnnMB_ValidatedInputs=KNNmb_out, 
                trainFilePath=trainEntryVar.get(),
                testFilePath=testEntryVar.get(),
                inProgressUpdateFunc=func_to_update_inProgress_text
            )
            master.after(1000, lambda processOut=processResultDict: update_success(processOut))
        except Exception as ex:
            master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))

    if len(WARNINGS) == 0:
        print("[KNN] VALIDATED_INPUTS: ", jsonDumps(KNNmb_out, indent=2))
        Thread(target=run_process).start()
    else:
        master.after(1000, lambda warnings=WARNINGS: update_failure(warnings))

def GB_MODEL_BUILD_SUBMIT (master:CTk, loading_gif_path:str, GBmb_inputs: dict, RFmb_resultsVar: StringVar, font: CTkFont, trainEntryVar: StringVar, testEntryVar: StringVar):
    
    inProgress = InProgressWindow(master, font, loading_gif_path)
    inProgress.create()

    def update_success (processOutput: dict):
        inProgress.destroy()
        RFmb_resultsVar.set(jsonDumps(processOutput, indent=4))
        CustomSuccessBox(master, "Calculations Completed !!", font)
        
    def update_failure (warnings: list):
        inProgress.destroy()
        RFmb_resultsVar.set('..')
        CustomWarningBox(master, warnings, font)
    
    GBmb_out = {k:v.get() for k,v in GBmb_inputs.items()}
    WARNINGS = []

    # FEATURES
    if not len(GBmb_out["FEATURES"]):
        master.after(1000, lambda warnings=['No FEATURES selected !!']: update_failure(warnings))
        return
    else:
        GBmb_out["FEATURES"] = GBmb_out["FEATURES"].split(',')

    # loss
    try:
        if GBmb_out["loss"].lower() in ['log_loss', 'exponential']:
            GBmb_out["loss"] = GBmb_out["loss"].lower()
        else:
            raise Exception()
    except:
        WARNINGS.append("loss must be an either of 'log_loss'/'exponential'")

    # learning_rate
    try:
        GBmb_out["learning_rate"] = float(GBmb_out["learning_rate"])
    except:
        WARNINGS.append('learning_rate must be float')

    # n_estimators
    try:
        GBmb_out["n_estimators"] = int(GBmb_out["n_estimators"])
    except:
        WARNINGS.append("n_estimators must be an INTEGER")

    # subsample
    try:
        GBmb_out["subsample"] = float(GBmb_out["subsample"])
    except:
        WARNINGS.append('subsample must be float')

    # criterion
    try:
        if GBmb_out["criterion"].lower() in ['friedman_mse', 'squared_error']:
            GBmb_out["criterion"] = GBmb_out["criterion"].lower()
        else:
            raise Exception()
    except:
        WARNINGS.append("criterion must be an either of 'log_loss'/'exponential'")

    # max_depth
    try:
        GBmb_out["max_depth"] = None if GBmb_out["max_depth"].lower()=='none' else int(GBmb_out["max_depth"])
    except:
        WARNINGS.append("max_depth must be an INTEGER or 'None'")

    # min_samples_split
    try:
        GBmb_out["min_samples_split"] = convertStrToIntOrFloat(GBmb_out["min_samples_split"])
    except:
        WARNINGS.append('min_samples_split must be int or float')
    
    # min_samples_leaf
    try:
        GBmb_out["min_samples_leaf"] = convertStrToIntOrFloat(GBmb_out["min_samples_leaf"])
    except:
        WARNINGS.append('min_samples_leaf must be int or float')

    # min_impurity_decrease
    try:
        GBmb_out["min_impurity_decrease"] = float(GBmb_out["min_impurity_decrease"])
    except:
        WARNINGS.append('min_impurity_decrease must be float')

    # min_weight_fraction_leaf
    try:
        GBmb_out["min_weight_fraction_leaf"] = float(GBmb_out["min_weight_fraction_leaf"])
    except:
        WARNINGS.append('min_weight_fraction_leaf must be float')

    # random_state
    try:
        GBmb_out["random_state"] = None if GBmb_out["random_state"].lower()=='none' else int(GBmb_out["random_state"])
    except:
        WARNINGS.append("random_state must be an INTEGER or 'None'")

    # warm start
    GBmb_out["warm_start"] = True if GBmb_out["warm_start"].lower()=='true' else False

    # max_features
    try:
        if GBmb_out["max_features"].lower()=='none':
            GBmb_out["max_features"] = None
        elif GBmb_out["max_features"].lower() in ['sqrt', 'log2']:
            GBmb_out["max_features"] = GBmb_out["max_features"].lower()
        else:
            GBmb_out["max_features"] = convertStrToIntOrFloat(GBmb_out["max_features"])
    except:
        WARNINGS.append("max_features must be an INTEGER, FLOAT or either of 'sqrt'/'log2'/None")
    
    # validation_fraction
    try:
        GBmb_out["validation_fraction"] = float(GBmb_out["validation_fraction"])
    except:
        WARNINGS.append('validation_fraction must be float')

    # tol
    try:
        GBmb_out["tol"] = float(GBmb_out["tol"])
    except:
        WARNINGS.append('tol must be float')
    
    # max_leaf_nodes
    # try:
    #     GBmb_out["max_leaf_nodes"] = None if GBmb_out["max_leaf_nodes"].lower()=='none' else int(GBmb_out["max_leaf_nodes"])
    # except:
    #     WARNINGS.append("max_leaf_nodes must be an INTEGER")

    # n_iter_no_change
    try:
        GBmb_out["n_iter_no_change"] = None if GBmb_out["n_iter_no_change"].lower()=='none' else int(GBmb_out["n_iter_no_change"])
    except:
        WARNINGS.append("n_iter_no_change must be an INTEGER or None")
    
    def func_to_update_inProgress_text(txt):
        inProgress.update_progress_verdict(txt)

    def run_process():
        try:
            processResultDict = GB_MODEL_BUILD_PROCESS(
                GbMB_ValidatedInputs=GBmb_out, 
                trainFilePath=trainEntryVar.get(),
                testFilePath=testEntryVar.get(),
                inProgressUpdateFunc=func_to_update_inProgress_text
            )
            master.after(1000, lambda processOut=processResultDict: update_success(processOut))
        except Exception as ex:
            master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))

    if len(WARNINGS) == 0:
        print("[GB] VALIDATED_INPUTS: ", jsonDumps(GBmb_out, indent=2))
        Thread(target=run_process).start()
    else:
        master.after(1000, lambda warnings=WARNINGS: update_failure(warnings))

def MLP_MODEL_BUILD_SUBMIT (master:CTk, loading_gif_path:str, MLPmb_inputs: dict, MLPmb_resultsVar: StringVar, font: CTkFont, trainEntryVar: StringVar, testEntryVar: StringVar):
    inProgress = InProgressWindow(master, font, loading_gif_path)
    inProgress.create()

    def update_success (processOutput: dict):
        inProgress.destroy()
        MLPmb_resultsVar.set(jsonDumps(processOutput, indent=4))
        CustomSuccessBox(master, "Calculations Completed !!", font)
        
    def update_failure (warnings: list):
        inProgress.destroy()
        MLPmb_resultsVar.set('..')
        CustomWarningBox(master, warnings, font)
    
    MLPmb_out = {k:v.get() for k,v in MLPmb_inputs.items()}
    WARNINGS = []

    # FEATURES
    if not len(MLPmb_out["FEATURES"]):
        master.after(1000, lambda warnings=['No FEATURES selected !!']: update_failure(warnings))
        return
    else:
        MLPmb_out["FEATURES"] = MLPmb_out["FEATURES"].split(',')
    
    # hidden_layer_sizes
    try:
        MLPmb_out["hidden_layer_sizes"] = tuple(
            ast_literal_eval(MLPmb_out["hidden_layer_sizes"])
        )
    except:
        WARNINGS.append("hidden_layer_sizes must be array-like of shape.\nThe ith element represents the number of neurons in the ith hidden layer.")
    
    # alpha
    try:
        MLPmb_out["alpha"] = float(MLPmb_out["alpha"])
    except:
        WARNINGS.append("alpha must be an FLOAT")

    # learning_rate_init
    try:
        MLPmb_out["learning_rate_init"] = float(MLPmb_out["learning_rate_init"])
    except:
        WARNINGS.append("learning_rate_init must be an FLOAT")
    
    # power_t
    try:
        MLPmb_out["power_t"] = float(MLPmb_out["power_t"])
    except:
        WARNINGS.append("power_t must be an FLOAT")
    
    # max_iter
    try:
        MLPmb_out["max_iter"] = int(MLPmb_out["max_iter"])
    except:
        WARNINGS.append("max_iter must be an INTEGER")

    # batch_size
    try:
        MLPmb_out["batch_size"] = 'auto' if MLPmb_out["batch_size"].lower()=='auto' else int(MLPmb_out["batch_size"])
    except:
        WARNINGS.append("batch_size must be an INTEGER or 'auto'")

    # random_state
    try:
        MLPmb_out["random_state"] = None if MLPmb_out["random_state"].lower()=='none' else int(MLPmb_out["random_state"])
    except:
        WARNINGS.append("random_state must be an INTEGER or 'None'")

    # tol
    try:
        MLPmb_out["tol"] = float(MLPmb_out["tol"])
    except:
        WARNINGS.append("tol must be an FLOAT")

    # momentum
    try:
        MLPmb_out["momentum"] = float(MLPmb_out["momentum"])
    except:
        WARNINGS.append("momentum must be an FLOAT")

    # validation_fraction
    try:
        MLPmb_out["validation_fraction"] = float(MLPmb_out["validation_fraction"])
    except:
        WARNINGS.append("validation_fraction must be an FLOAT")

    # n_iter_no_change
    try:
        MLPmb_out["n_iter_no_change"] = int(MLPmb_out["n_iter_no_change"])
    except:
        WARNINGS.append("n_iter_no_change must be an INTEGER")
    
    # max_fun
    try:
        MLPmb_out["max_fun"] = int(MLPmb_out["max_fun"])
    except:
        WARNINGS.append("max_fun must be an INTEGER")

    MLPmb_out["shuffle"] = bool(MLPmb_out["shuffle"])
    MLPmb_out["verbose"] = bool(MLPmb_out["verbose"])
    MLPmb_out["warm_start"] = bool(MLPmb_out["warm_start"])
    MLPmb_out["nesterovs_momentum"] = bool(MLPmb_out["nesterovs_momentum"])
    MLPmb_out["early_stopping"] = bool(MLPmb_out["early_stopping"])
    
    def func_to_update_inProgress_text(txt):
        inProgress.update_progress_verdict(txt)

    def run_process():
        try:
            processResultDict = MLP_MODEL_BUILD_PROCESS(
                MlpMB_ValidatedInputs=MLPmb_out, 
                trainFilePath=trainEntryVar.get(),
                testFilePath=testEntryVar.get(),
                inProgressUpdateFunc=func_to_update_inProgress_text
            )
            master.after(1000, lambda processOut=processResultDict: update_success(processOut))
        except Exception as ex:
            master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))

    if len(WARNINGS) == 0:
        print("[MLP] VALIDATED_INPUTS: ", jsonDumps(MLPmb_out, indent=2))
        Thread(target=run_process).start()
    else:
        master.after(1000, lambda warnings=WARNINGS: update_failure(warnings))
