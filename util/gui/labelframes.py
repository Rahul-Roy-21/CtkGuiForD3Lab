import customtkinter as ctk
from tkinter import LabelFrame as tkLabelFrame
from data import DATA
from util.gui.widgets import *
COLORS = DATA['colors']

# With algoMap = {'RF': {'algo_name': 'Random Forest', 'algo_hp_optim_func': RF_HP_OPTIM}}
# For RF: algoValueInMap = algoMap['RF']
# _DATA[or fields_data]: field configurations
class HyperParamOptim_AlgoLabelFrame:
    def __init__(self, master_panel: ctk.CTkFrame, algoValueInMap: dict, algo_hp_optim_fields_data: dict, my_font: ctk.CTkFont, fg_color: str, result_Loading_ImgPath: str, SELECTED_FEATURES:ctk.StringVar, trainVar:ctk.StringVar, testVar:ctk.StringVar, hyperParamsFrame_NumOfCells: int):
        self.master = master_panel
        self.result_Loading_ImgPath = result_Loading_ImgPath
        self.my_font = my_font
        self.fg_color = fg_color
        self._DATA = algo_hp_optim_fields_data
        self.TRAIN_VAR, self.TEST_VAR = trainVar, testVar
        self.hyperParamsFrame_NumOfCells = hyperParamsFrame_NumOfCells
        self.algo_hp_optim_func_onSubmit = algoValueInMap['algo_hp_optim_func'] # Call this func in the submit along with lambda and arguments
    
        self.algo_labelFrame = self._get_labelframe(master_panel, algoValueInMap['algo_name'])
        self.algo_labelFrame.grid_columnconfigure(tuple(range(9)), weight=1)
        self.algo_labelFrame.grid_rowconfigure(2, weight=1) # Results

        # Build Method[4 cols], Scoring [4 cols], Cross-Valid-Folds [1 cols]
        self.algo_method = self._get_labelframe(self.algo_labelFrame, 'Method')
        self.algo_scoring = self._get_labelframe(self.algo_labelFrame, 'Scoring')
        self.algo_cvfolds = self._get_labelframe(self.algo_labelFrame, 'Cross-Valid Folds')
        self.algo_hp_optim = self._get_labelframe(self.algo_labelFrame, 'HyperParameters')
        self.algo_results = self._get_labelframe(self.algo_labelFrame, 'Results')

        self.algo_method.grid(row=0,column=0,columnspan=4,padx=5,pady=2,sticky=ctk.NSEW)
        self.algo_scoring.grid(row=0,column=4,columnspan=4,padx=5,pady=2,sticky=ctk.NSEW)
        self.algo_cvfolds.grid(row=0,column=8,columnspan=1,padx=5,pady=2,sticky=ctk.NSEW)
        self.algo_hp_optim.grid(row=1,column=0,columnspan=9,padx=5,pady=2,sticky=ctk.NSEW)
        self.algo_results.grid(row=2,column=0,columnspan=9,padx=5,pady=1,sticky=ctk.NSEW)

        self.algo_inputs = {'FEATURES':SELECTED_FEATURES}
        self._build_method_scoring_cvfolds_row()
        self._CREATE_HP_OPTIM_FIELDS(self._DATA['hp_optim_field_configs'])
        self.algo_results_var = self._build_results_row()

    def _GET_ALGO_LABELFRAME (self):
        return self.algo_labelFrame

    def _CREATE_HP_OPTIM_FIELDS (self, field_configs: dict):
        self.algo_hp_optim.grid_columnconfigure(tuple(range(self.hyperParamsFrame_NumOfCells)), weight=1)
        self.maxRow = 0
        # For RF - 2
        # DEMO_FIELD_CONFIGS = {'field_name': {'type':FieldVar, 'grid': {'row','col','colspan'}}..}
        for _field_name, FieldTypeAndGrid in field_configs.items():
            _field, _grid = FieldTypeAndGrid['type'], FieldTypeAndGrid['grid']

            _field_lbFrame = self._get_labelframe(self.algo_hp_optim, _field_name)
            _field_lbFrame.grid(row=_grid['row'], column=_grid['col'], columnspan=_grid['colspan'], padx=10, pady=5, sticky=ctk.EW)
            self.maxRow = max(self.maxRow, _grid['row']) # For Submit btn's ROW NUMBER
            _field_vars = _field._get_vars()

            if type(_field) == MyStepRangeEntryField:
                _field_entry = MyStepRangeEntry(
                    parent=_field_lbFrame,
                    from_var=_field_vars['from_var'],
                    to_var=_field_vars['to_var'],
                    step_var=_field_vars['step_var'],
                    my_font=self.my_font,
                    MIN_VAL=int(_field_vars['min_val']),
                    MAX_VAL=int(_field_vars['max_val']),
                    MAX_STEPS=int(_field_vars['max_steps'])
                )
                self.algo_inputs[_field_name] = {
                    '_FROM':_field_vars['from_var'], 
                    '_TO':_field_vars['to_var'], 
                    '_STEP':_field_vars['step_var']
                }

            elif type(_field) == MyRangeEntryField:
                _field_entry = MyRangeEntry(
                    parent=_field_lbFrame,
                    from_var=_field_vars['from_var'],
                    to_var=_field_vars['to_var'],
                    my_font=self.my_font,
                    MIN_VAL=int(_field_vars['min_val']),
                    MAX_VAL=int(_field_vars['max_val']),
                )
                self.algo_inputs[_field_name] = {
                    '_FROM':_field_vars['from_var'], '_TO':_field_vars['to_var']
                }

            elif type(_field) == MyLogarithmicRangeEntryField:
                _field_entry = MyFloatingLogRangeEntry(
                    parent=_field_lbFrame,
                    from_var=_field_vars['from_var'],
                    to_var=_field_vars['to_var'],
                    my_font=self.my_font,
                    MIN_VAL=int(_field_vars['min_val']),
                    MAX_VAL=int(_field_vars['max_val']),
                )
                self.algo_inputs[_field_name] = {
                    '_FROM':_field_vars['from_var'], '_TO':_field_vars['to_var']
                }

            elif type(_field) == MultiSelectEntryField:
                _field_entry = MultiSelectEntry(
                    parent=_field_lbFrame,
                    whatToChoosePlural=_field_name,
                    my_font=self.my_font,
                    tkVar=_field_vars['selected_opt_var'],
                    options=_field_vars['options']
                )
                self.algo_inputs[_field_name] = _field_vars['selected_opt_var']

            _field_entry.grid(row=0, column=0, padx=10, pady=5, sticky=ctk.EW)

        # Create Submit Button
        submit_btn = CREATE_SUBMIT_BUTTON(
            master_frame=self.algo_hp_optim, 
            my_font=self.my_font, 
            commandOnSubmit= lambda: self.algo_hp_optim_func_onSubmit(self.master, self.result_Loading_ImgPath, self.algo_inputs, self.algo_results_var, self.my_font, self.TRAIN_VAR, self.TEST_VAR)
        )
        submit_btn.grid(row=self.maxRow+1, column=0, columnspan=self.hyperParamsFrame_NumOfCells, padx=10, pady=10)


    def _build_method_scoring_cvfolds_row(self):
        self.algo_method.grid_columnconfigure(0, weight=1)
        self.algo_scoring.grid_columnconfigure(0, weight=1)
        self.algo_cvfolds.grid_columnconfigure(0, weight=1)

        method_optionmenu = CREATE_OPTIONMENU(
            master_frame=self.algo_method, my_font=self.my_font, 
            all_opts=self._DATA['method_opts'], 
            selected_optVar=self._DATA['method_selected_optVar'],
        )
        scoring_optionmenu = CREATE_OPTIONMENU(
            master_frame=self.algo_scoring, my_font=self.my_font, 
            all_opts=self._DATA['scoring_opts'], 
            selected_optVar=self._DATA['scoring_selected_optVar'],
        )
        cvfolds_integer_entry = MyIntegerEntry(
            parent=self.algo_cvfolds, my_font=self.my_font, 
            tkVar=self._DATA['cvfolds_entryVar'], 
            min_value=int(self._DATA['cvfolds_min']),
            max_value=int(self._DATA['cvfolds_max'])
        )

        self.algo_inputs['METHOD'] = self._DATA['method_selected_optVar']
        self.algo_inputs['SCORING'] = self._DATA['scoring_selected_optVar']
        self.algo_inputs['CROSS_FOLD_VALID'] = self._DATA['cvfolds_entryVar']

        method_optionmenu.grid(row=0, column=0, padx=10, pady=5, sticky=ctk.EW)
        scoring_optionmenu.grid(row=0, column=0, padx=10, pady=5, sticky=ctk.EW)
        cvfolds_integer_entry.grid(row=0, column=0, padx=10, pady=5, sticky=ctk.EW)

    def _build_results_row(self):
        self.algo_results.grid_columnconfigure(0, weight=1)
        self.algo_results.grid_rowconfigure(0, weight=1)
        self.algo_results_var = StringVar(value="...")
        RF_resultTextBox = SyncableTextBox(
            master=self.algo_results,
            text_variable=self.algo_results_var,
            my_font=self.my_font
        )
        RF_resultTextBox.grid(row=0, column=0, padx=10, pady=5, sticky=ctk.NSEW)
        return self.algo_results_var

    def _get_labelframe (self, master_frame: ctk.CTkFrame, label_txt: str):
        return tkLabelFrame(
            master=master_frame, text=label_txt, font=self.my_font, 
            labelanchor=ctk.NW, background=self.fg_color
        )
    

class ModelBuild_AlgoLabelFrame:
    def __init__(self, master_panel: ctk.CTkFrame, algoValueInMap: dict, algo_model_build_fields_data: dict, my_font: ctk.CTkFont, fg_color: str, result_Loading_ImgPath: str, SELECTED_FEATURES:ctk.StringVar, trainVar:ctk.StringVar, testVar:ctk.StringVar, hyperParamsFrame_NumOfCells: int):
        self.master = master_panel
        self.result_Loading_ImgPath = result_Loading_ImgPath
        self.my_font = my_font
        self.fg_color = fg_color
        self._DATA = algo_model_build_fields_data
        self.TRAIN_VAR, self.TEST_VAR = trainVar, testVar
        self.hyperParamsFrame_NumOfCells = hyperParamsFrame_NumOfCells
        self.algo_model_build_func_onSubmit = algoValueInMap['algo_model_build_func'] # Call this func in the submit along with lambda and arguments
    
        self.algo_labelFrame = self._get_labelframe(master_panel, algoValueInMap['algo_name'])
        self.algo_labelFrame.grid_columnconfigure(0, weight=1)
        self.algo_labelFrame.grid_rowconfigure(1, weight=1) # Results

        self.algo_hp_optim = self._get_labelframe(self.algo_labelFrame, 'HyperParameters')
        self.algo_results = self._get_labelframe(self.algo_labelFrame, 'Results')

        self.algo_hp_optim.grid(row=0,column=0,columnspan=1,padx=5,pady=2,sticky=ctk.NSEW)
        self.algo_results.grid(row=1,column=0,columnspan=1,padx=5,pady=2,sticky=ctk.NSEW)

        self.algo_inputs = {'FEATURES':SELECTED_FEATURES}
        self._CREATE_MODEL_BUILD_FIELDS(self._DATA['model_build_field_configs'])
        self.algo_results_var = self._build_results_row()

    def _GET_ALGO_LABELFRAME (self):
        return self.algo_labelFrame

    def _CREATE_MODEL_BUILD_FIELDS (self, field_configs: dict):
        self.algo_hp_optim.grid_columnconfigure(tuple(range(self.hyperParamsFrame_NumOfCells)), weight=1)
        self.maxRow = 0
        # For RF - 2
        # DEMO_FIELD_CONFIGS = {'field_name': {'type': 'StringVar/CtkOptionMenu', 'grid': {'row','col','colspan'}}..}
        for _field_name, FieldTypeAndGrid in field_configs.items():
            _field, _grid = FieldTypeAndGrid['type'], FieldTypeAndGrid['grid']

            _field_lbFrame = self._get_labelframe(self.algo_hp_optim, _field_name)
            _field_lbFrame.grid(row=_grid['row'], column=_grid['col'], columnspan=_grid['colspan'], padx=10, pady=5, sticky=ctk.EW)
            self.maxRow = max(self.maxRow, _grid['row']) # For Submit btn's ROW NUMBER
            _field_vars = _field._get_vars()

            # For StringVar: vars=[textVariable, defaultVal], For OptionMenu values=[textVariable, options, defaultOpt] 
            if type(_field) == StringField:
                _field_entry = CREATE_ENTRY(
                    master_frame=_field_lbFrame,
                    my_font=self.my_font,
                    textVariable=_field_vars['var']
                )
                _field_entry.grid(row=0, column=0, padx=10, pady=5, sticky=ctk.EW)
                self.algo_inputs[_field_name] = _field_vars['var']

            elif type(_field) == OptionMenuField:
                _field_entry = CREATE_OPTIONMENU(
                    master_frame=_field_lbFrame,
                    my_font=self.my_font,
                    all_opts=_field_vars['options'],
                    selected_optVar=_field_vars['var']
                )
                _field_entry.grid(row=0, column=0, padx=10, pady=5, sticky=ctk.EW)
                self.algo_inputs[_field_name] = _field_vars['var']

        # Create Submit Button
        submit_btn = CREATE_SUBMIT_BUTTON(
            master_frame=self.algo_hp_optim, 
            my_font=self.my_font, 
            commandOnSubmit= lambda: self.algo_model_build_func_onSubmit(self.master, self.result_Loading_ImgPath, self.algo_inputs, self.algo_results_var, self.my_font, self.TRAIN_VAR, self.TEST_VAR)
        )
        submit_btn.grid(row=self.maxRow+1, column=0, columnspan=self.hyperParamsFrame_NumOfCells, padx=10, pady=10)

    def _build_results_row(self):
        self.algo_results.grid_columnconfigure(0, weight=1)
        self.algo_results.grid_rowconfigure(0, weight=1)
        self.algo_results_var = StringVar(value="...")
        RF_resultTextBox = SyncableTextBox(
            master=self.algo_results,
            text_variable=self.algo_results_var,
            my_font=self.my_font
        )
        RF_resultTextBox.grid(row=0, column=0, padx=10, pady=5, sticky=ctk.NSEW)
        return self.algo_results_var

    def _get_labelframe (self, master_frame: ctk.CTkFrame, label_txt: str):
        return tkLabelFrame(
            master=master_frame, text=label_txt, font=self.my_font, 
            labelanchor=ctk.NW, background=self.fg_color
        )




# HP_OPTIM FIELDS
class MyStepRangeEntryField:
    def __init__(self, from_var:ctk.IntVar, to_var:ctk.IntVar, step_var:ctk.IntVar, min_val:int, max_val:int, max_steps:int):
        self.vars = {
            'from_var':from_var, 'to_var':to_var, 'step_var':step_var,
            'min_val': min_val, 'max_val':max_val, 'max_steps': max_steps
        }
    def _get_vars(self):
        return self.vars
        
class MultiSelectEntryField:
    def __init__(self, options:list, selected_opt_var: ctk.StringVar, min_choice:int = 2):
        self.vars = {'options':options, 'selected_opt_var':selected_opt_var, 'min_choice':min_choice}
    def _get_vars(self):
        return self.vars

class MyRangeEntryField:
    def __init__(self, from_var:ctk.IntVar, to_var:ctk.IntVar, min_val:int, max_val:int):
        self.vars = {'from_var':from_var, 'to_var':to_var, 'min_val': min_val, 'max_val':max_val}
    def _get_vars(self):
        return self.vars

class MyLogarithmicRangeEntryField:
    def __init__(self, from_var:ctk.DoubleVar, to_var:ctk.DoubleVar, min_val:float, max_val:float):
        self.vars = {'from_var':from_var, 'to_var':to_var, 'min_val': min_val, 'max_val':max_val}
    def _get_vars(self):
        return self.vars

# MODEL_BUILD_FIELDS  
class StringField:
    def __init__(self, variable_withDefaultVal:ctk.StringVar):
        self.vars = {'var':variable_withDefaultVal}
    def _get_vars(self):
        return self.vars

class OptionMenuField:
    def __init__(self, variable_withSelectedOpt:ctk.StringVar, options:list[str]):
        self.vars = {'var':variable_withSelectedOpt, 'options':options}
    def _get_vars(self):
        return self.vars


def CREATE_ENTRY(master_frame, my_font, textVariable:ctk.StringVar) -> ctk.CTkEntry:
    return ctk.CTkEntry(
        master=master_frame, 
        font=my_font,
        textvariable=textVariable,
        corner_radius=0,
        border_width=0,
    )

def CREATE_OPTIONMENU (master_frame, my_font, all_opts:list, selected_optVar:ctk.StringVar) -> ctk.CTkOptionMenu:
    return ctk.CTkOptionMenu(
        master=master_frame, 
        values=all_opts,
        variable=selected_optVar,
        font=my_font,
        dropdown_font=my_font,
        corner_radius=0,
        button_color=COLORS['GREY_FG'],
        button_hover_color=COLORS['GREY_FG'],
        fg_color=COLORS['GREY_FG']
    )

def CREATE_SUBMIT_BUTTON(master_frame:ctk.CTkFrame, my_font:str, commandOnSubmit) -> ctk.CTkButton:
    return CTkButton(
        master=master_frame,
        text='Submit',
        font=my_font,
        fg_color=COLORS['MEDIUMGREEN_FG'],
        hover_color=COLORS['MEDIUMGREEN_HOVER_FG'],
        text_color='white',
        corner_radius=0,
        width=300,
        border_spacing=0,
        command=commandOnSubmit
    )