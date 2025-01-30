import customtkinter as ctk
from PIL import Image
from tkinter import filedialog
import json
from util.ml.functions import CHECK_XLS_FILES, GET_RANKED_FEATURES
from util.gui.widgets import CustomWarningBox, FeatureSelectEntry

class TaskPanel:
    def __init__(self,master:ctk.CTk, my_font:ctk.CTkFont, 
                 fg_color: str, taskSelectBtns_fromSidePanel: dict):
        self.master = master
        self.my_font = my_font
        self.fg_color = fg_color
        self.task_panels = self._create_widgets()
        self.task_select_btns = taskSelectBtns_fromSidePanel
        self._CONFIGURE_SIDE_PANEL_BTNS_TO_SHOW_TASK_PANEL(self.task_select_btns, self.task_panels)
    
    def _get_task_panels(self):
        return self.task_panels

    def _create_widgets(self) -> dict[str, ctk.CTkFrame]:
        hyperparam_optim_panel=ctk.CTkFrame(master=self.master, fg_color=self.fg_color)
        model_build_panel=ctk.CTkFrame(master=self.master, fg_color=self.fg_color)
        default_panel=ctk.CTkFrame(master=self.master, fg_color=self.fg_color)

        # Configure responsiveness for child-panels
        hyperparam_optim_panel.grid_columnconfigure(0,weight=1)
        hyperparam_optim_panel.grid_rowconfigure(tuple(range(1,7)), weight=1)
        model_build_panel.grid_columnconfigure(0,weight=1)
        model_build_panel.grid_rowconfigure(tuple(range(1,7)), weight=1)
        default_panel.grid_columnconfigure(0,weight=1)

        default_panel_label=ctk.CTkLabel(master=default_panel,text='default_panel')
        default_panel_label.grid(row=0,column=0)

        hyperparam_optim_label=ctk.CTkLabel(master=hyperparam_optim_panel,text='hp_optim_panel')
        hyperparam_optim_label.grid(row=0,column=0)
        model_build_label=ctk.CTkLabel(master=model_build_panel,text='model_build_panel')
        model_build_label.grid(row=0,column=0)

        return {'hp_optim': hyperparam_optim_panel, 'model_build': model_build_panel, 'default': default_panel}

    def _CONFIGURE_SIDE_PANEL_BTNS_TO_SHOW_TASK_PANEL(self, task_select_btns: dict[str, ctk.CTkButton], task_panels: dict[str, ctk.CTkFrame]):
        '''Configures task_select_btns [from side-panel] to SHOW_FRAME(..) their respective panels'''
        for TASK in task_select_btns.keys():
            task_select_btns[TASK].configure(
                command = lambda TASK_NAME=TASK: self.show_frame(task_panels, task_select_btns, TASK_NAME)
            )
        self.show_frame(task_panels, task_select_btns, 'default')

    def show_frame(self, task_panels: dict[str, ctk.CTkFrame], task_btns: dict[str, ctk.CTkButton], TASK: str):
        for panel in task_panels.values():
            panel.grid_forget()
        for btn in task_btns.values():
            btn.configure(border_width=0)
            btn.configure(text=btn.cget('text').rstrip('*'))

        # Show selected frame and * in btn Text
        task_panels[TASK].grid(row=2,column=1,rowspan=9,columnspan=7,sticky=ctk.NSEW,padx=(2,5),pady=5)
        if TASK.lower() != 'default':
            task_btns[TASK].configure(border_width=3, border_color='#000')
            task_btns[TASK].configure(text=task_btns[TASK].cget('text')+'*')

class SidePanel:
    def __init__(self, master:ctk.CTk, my_font:ctk.CTkFont, colors_fgAndBtns: dict, img_pathsAndSizes: dict):
        self.master = master
        self.my_font = my_font
        # img_pathsAndSizes = {'hp_optim': {'path', 'size'}, 'model_build': {'path', 'size'}}
        self.img_pathsAndSizes = img_pathsAndSizes
        # colors = {'fg', 'btns': {'hp_optim': {'fg', 'hover'}, 'model_build': {'fg', 'hover'}}}
        self.colors = colors_fgAndBtns

        self.side_panel = ctk.CTkFrame(
            master=self.master,
            fg_color=self.colors['fg']
        )
        self.side_panel.grid(row=2,column=0,rowspan=9,sticky=ctk.NSEW,padx=(5,2),pady=5)
        self.task_panel_btns = self.create_widgets() # Initialize components

    def create_widgets(self) -> dict[str, ctk.CTkButton]:
        hp_optim_img = ctk.CTkImage(
            light_image=Image.open(self.img_pathsAndSizes['hp_optim']['path']), 
            size=self.img_pathsAndSizes['hp_optim']['size']
        )
        model_build_img = ctk.CTkImage(
            light_image=Image.open(self.img_pathsAndSizes['model_build']['path']), 
            size=self.img_pathsAndSizes['model_build']['size']
        )

        hyperparam_optim_btn = self.create_panelSelectionButton (
            btn_text="HYPERPARAMETER\nOPTIMIZATION", 
            btn_image=hp_optim_img,
            BTN_COLORS=self.colors['btns']['hp_optim']
        )
        model_build_btn = self.create_panelSelectionButton (
            btn_text="MODEL\nBUILD", 
            btn_image=model_build_img,
            BTN_COLORS=self.colors['btns']['model_build']
        )
        hyperparam_optim_btn.grid(row=0,column=0,rowspan=3,sticky=ctk.NSEW, padx=10,pady=5)
        model_build_btn.grid(row=3,column=0,rowspan=3,sticky=ctk.NSEW, padx=10,pady=5)

        return {'hp_optim': hyperparam_optim_btn, 'model_build': model_build_btn}
    
    def GET_SELECT_TASK_BTNS (self) -> dict:
        return self.task_panel_btns

    def create_panelSelectionButton(self, btn_text:str, btn_image: ctk.CTkImage, BTN_COLORS: dict):
        button = ctk.CTkButton(
            master=self.side_panel,
            text=btn_text, 
            image=btn_image,
            compound=ctk.LEFT,
            font=self.my_font,
            fg_color=BTN_COLORS['fg'],
            hover_color=BTN_COLORS['hover'],
            text_color='white',
            corner_radius=0
        )
        return button

class DataSetPanel:
    def __init__(self, master:ctk.CTk, LOADED_FEATURES, SELECTED_FEATURES, 
             train_entryVar: ctk.StringVar, test_entryVar: ctk.StringVar,
             my_font:ctk.CTkFont, colors_fgAndBtns: dict, img_pathsAndSizes: dict):
        
        self.master = master
        self.ALL_LOADED_FEATURES = LOADED_FEATURES
        self.SELECTED_FEATURES = SELECTED_FEATURES
        self.train_entryVar = train_entryVar
        self.test_entryVar = test_entryVar
        self.my_font = my_font
        
        # img_pathsAndSizes = {'logo': {'path': os.path.join(..), 'size':(65,65)}, 'upload': {'path': os.path.join(..), 'size': (20,20)}}
        self.img_pathsAndSizes = img_pathsAndSizes
        # colors = {'fg': $, 'btn': {'fg': $, 'hover': $}}
        self.colors = colors_fgAndBtns

        self.dataset_frame = ctk.CTkFrame(
            master=self.master,
            fg_color=self.colors['fg']
        )

        # Set up grid configuration
        self.setup_grid()
        # Initialize components
        self.create_widgets()

    def setup_grid(self, row=0, col=0, colspan=8):
        # 8 cols: 1=label, 3=entry, 1=btn, 2=<SPACE>, 1=Logo
        self.dataset_frame.grid_columnconfigure((1, 2, 3), weight=3)
        self.dataset_frame.grid_columnconfigure((5, 6), weight=1)

        # Place the dataset frame into root
        self.dataset_frame.grid(row=row, column=col, columnspan=colspan, sticky=ctk.EW, padx=5, pady=(7, 0))
    
    def create_widgets(self):
        # Create labels, entries, and buttons
        self.train_label = ctk.CTkLabel(master=self.dataset_frame, text='TRAIN', font=self.my_font)
        self.test_label = ctk.CTkLabel(master=self.dataset_frame, text='TEST', font=self.my_font)

        # Create entries for Train,Test
        self.train_entry = self.create_entry(self.train_entryVar)
        self.test_entry = self.create_entry(self.test_entryVar)

        # Upload button for Train, Test
        self.train_btn = self.create_upload_button('Train', self.train_entryVar)
        self.test_btn = self.create_upload_button('Test', self.test_entryVar)

        # Create logo
        self.logo_img = ctk.CTkImage(
            light_image=Image.open(self.img_pathsAndSizes['logo']['path']), 
            size=self.img_pathsAndSizes['logo']['size']
        )
        self.logo_label = ctk.CTkLabel(master=self.dataset_frame, image=self.logo_img, text='')
        self.logo_label.grid(row=0, column=7, rowspan=2, padx=7, pady=7, sticky=ctk.NSEW)

        # Place widgets in grid
        self.train_label.grid(row=0, column=0, padx=5, pady=5, sticky=ctk.EW)
        self.train_entry.grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky=ctk.EW)
        self.train_btn.grid(row=0, column=4, padx=5, pady=5, sticky=ctk.EW)

        self.test_label.grid(row=1, column=0, padx=5, pady=5, sticky=ctk.EW)
        self.test_entry.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky=ctk.EW)
        self.test_btn.grid(row=1, column=4, padx=5, pady=5, sticky=ctk.EW)

    def create_entry(self, entry_var):
        entry = ctk.CTkEntry(
            master=self.dataset_frame,
            textvariable=entry_var,
            border_width=0,
            corner_radius=0,
            font=self.my_font
        )
        entry.bind('<FocusIn>', lambda e: self.on_focus_in(entry))
        entry.bind('<FocusOut>', lambda e: self.on_focus_out(entry))
        return entry

    def create_upload_button(self, field_name, file_path_entry_var):
        upload_img = ctk.CTkImage(
            light_image=Image.open(self.img_pathsAndSizes['upload']['path']), 
            size=self.img_pathsAndSizes['upload']['size']
        )
        button = ctk.CTkButton(
            master=self.dataset_frame,
            text='Upload',
            compound='left',
            image=upload_img,
            font=self.my_font,
            fg_color=self.colors['btn']['fg'],
            hover_color=self.colors['btn']['hover'],
            text_color='white',
            corner_radius=0,
            width=100,
            border_spacing=0,
            command=lambda: self.open_file_dialog(field_name, file_path_entry_var)
        )
        return button

    def on_focus_in(self, ctk_entry):
        ctk_entry.configure(border_color="#111")

    def on_focus_out(self, ctk_entry):
        ctk_entry.configure(border_color="#bbb")

    def open_file_dialog(self, field_name, file_path_entry_var):
        file_path = filedialog.askopenfilename(
            title=f"Select a file for {field_name}",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if file_path:
            file_path_entry_var.set(file_path)
            self.validate_train_test_files()

    def validate_train_test_files(self):
        if not self.train_entryVar.get() or not self.test_entryVar.get():
            return
        valid, loaded_columns = CHECK_XLS_FILES(self.train_entryVar.get(), self.test_entryVar.get())
        if not valid:
            self.ALL_LOADED_FEATURES.set("")
            self.SELECTED_FEATURES.set("")
            CustomWarningBox(
                parent=self.master, my_font=self.my_font, 
                warnings=["Train and Test files do not have identical column sets."]
            )
            return
        
        ranked_features_dict = GET_RANKED_FEATURES(self.train_entryVar.get())
        self.ALL_LOADED_FEATURES.set(json.dumps(ranked_features_dict))
        
        selected_features = [v['Feature'] for k,v in ranked_features_dict.items() if 1<=k<=10]
        self.SELECTED_FEATURES.set(",".join(selected_features))

class FeatureAndAlgorithmFrame:
    def __init__(self, masterFrame: ctk.CTkFrame, my_font: ctk.CTkFont, colors: dict, 
                LOADED_FEATURES: ctk.StringVar, SELECTED_FEATURES: ctk.StringVar, 
                LIST_OF_ALGORITHMS: list, SELECTED_ALGORITHM: ctk.StringVar,
                TRAIN_FILE_PATH: ctk.StringVar, TEST_FILE_PATH: ctk.StringVar):
        self.master = masterFrame
        self.my_font = my_font
        self.colors = colors
        self.loaded_features = LOADED_FEATURES
        self.selected_features = SELECTED_FEATURES
        self.list_of_algos = LIST_OF_ALGORITHMS
        self.selected_algo = SELECTED_ALGORITHM
        self.train_entryVar = TRAIN_FILE_PATH
        self.test_entryVar = TEST_FILE_PATH

        self._create_widgets()
    
    def _create_widgets(self):
        self.featureAlgo_frame=ctk.CTkFrame(
            master=self.master, fg_color=self.colors['fg']
        )
        self.featureAlgo_frame.grid(row=0,column=0,columnspan=7,sticky=ctk.NSEW,padx=(5,2),pady=5)
        self.featureAlgo_frame.grid_columnconfigure((1,2,3,5,6), weight=1)
        # ROW: featureLabel(0)--featureMultiSelectEntry(1,2,3)--algoLabel(4)--algoDropDown(5,6)

        self.features_label=ctk.CTkLabel(master=self.featureAlgo_frame, text='Features:', font=self.my_font)
        self.features_multiSelectEntry = FeatureSelectEntry(
            parent=self.featureAlgo_frame,
            my_font=self.my_font,
            selectedOptionsVar=self.selected_features,
            allOptionsVar=self.loaded_features,
            trainPathVar=self.train_entryVar,
            testPathVar=self.test_entryVar
        )

        self.algo_label=ctk.CTkLabel(master=self.featureAlgo_frame, text='Algorithm:', font=self.my_font)
        self.algo_dropdown=ctk.CTkOptionMenu(
            master=self.featureAlgo_frame, 
            values=self.list_of_algos,
            font=self.my_font,
            dropdown_font=self.my_font,
            variable=self.selected_algo,
            corner_radius=0,
            button_color=self.colors['btn']['fg'],
            button_hover_color=self.colors['btn']['hover'],
            fg_color=self.colors['btn']['fg']
        )
        
        self.features_label.grid(row=0,column=0,padx=5,pady=5,sticky=ctk.NSEW)
        self.features_multiSelectEntry.grid(row=0,column=1,columnspan=3,padx=5,pady=5,sticky=ctk.NSEW)
        self.algo_label.grid(row=0,column=4,padx=5,pady=5,sticky=ctk.NSEW)
        self.algo_dropdown.grid(row=0,column=5,columnspan=2,padx=5,pady=5,sticky=ctk.NSEW)
    
    # algo_dropdown needed later to configure the show_frames() function
    def _get_algorithm_optionmenu (self):
        return self.algo_dropdown