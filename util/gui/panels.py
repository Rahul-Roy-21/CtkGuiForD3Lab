import customtkinter as ctk
from PIL import Image
from tkinter import filedialog, LabelFrame as tkLabelFrame
import json
from threading import Thread
from util.gui.widgets import getImgPath
from util.ml.functions import CHECK_XLS_FILES, GET_RANKED_FEATURES, KENNARD_STONE, ACTIVITY_BASED_DIV, RANDOM_DIV
from util.gui.widgets import CustomWarningBox, FeatureSelectEntry, InProgressWindow, CustomSuccessBox
from data import _COMMON_PROPS

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
        dataset_div_panel=ctk.CTkFrame(master=self.master, fg_color=self.fg_color)
        settings_panel=ctk.CTkFrame(master=self.master, fg_color=self.fg_color)
        default_panel=ctk.CTkFrame(master=self.master, fg_color=self.fg_color)

        # Configure responsiveness for child-panels
        hyperparam_optim_panel.grid_columnconfigure(0,weight=1)
        hyperparam_optim_panel.grid_rowconfigure(tuple(range(1,7)), weight=1)
        model_build_panel.grid_columnconfigure(0,weight=1)
        model_build_panel.grid_rowconfigure(tuple(range(1,7)), weight=1)
        dataset_div_panel.grid_columnconfigure(0,weight=1)
        dataset_div_panel.grid_rowconfigure(tuple(range(1,7)), weight=1)
        settings_panel.grid_columnconfigure(0,weight=1)
        settings_panel.grid_rowconfigure(tuple(range(1,7)), weight=1)
        default_panel.grid_columnconfigure(0,weight=1)

        default_panel_label=ctk.CTkLabel(master=default_panel,text='default_panel')
        default_panel_label.grid(row=0,column=0)
        hyperparam_optim_label=ctk.CTkLabel(master=hyperparam_optim_panel,text='hp_optim_panel')
        hyperparam_optim_label.grid(row=0,column=0)
        model_build_label=ctk.CTkLabel(master=model_build_panel,text='model_build_panel')
        model_build_label.grid(row=0,column=0)
        dataset_div_label=ctk.CTkLabel(master=dataset_div_panel,text='dataset_div_panel')
        dataset_div_label.grid(row=0,column=0)
        settings_label=ctk.CTkLabel(master=settings_panel,text='settings_panel')
        settings_label.grid(row=0,column=0)

        return {'hp_optim': hyperparam_optim_panel, 'model_build': model_build_panel, 'default': default_panel, 'settings': settings_panel, 'dataset_div': dataset_div_panel}

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
        self.side_panel.grid(row=2,column=0,rowspan=12,sticky=ctk.NSEW,padx=(5,2),pady=5)
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
        dataset_div_img = ctk.CTkImage(
            light_image=Image.open(self.img_pathsAndSizes['dataset_div']['path']), 
            size=self.img_pathsAndSizes['dataset_div']['size']
        )
        settings_img = ctk.CTkImage(
            light_image=Image.open(self.img_pathsAndSizes['settings']['path']), 
            size=self.img_pathsAndSizes['settings']['size']
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
        dataset_div_btn = self.create_panelSelectionButton (
            btn_text="DATASET\nDIVISION", 
            btn_image=dataset_div_img,
            BTN_COLORS=self.colors['btns']['dataset_div']
        )
        settings_btn = self.create_panelSelectionButton (
            btn_text="SETTINGS", 
            btn_image=settings_img,
            BTN_COLORS=self.colors['btns']['settings']
        )
    
        hyperparam_optim_btn.grid(row=0,column=0,rowspan=3,sticky=ctk.NSEW, padx=10,pady=5)
        model_build_btn.grid(row=3,column=0,rowspan=3,sticky=ctk.NSEW, padx=10,pady=5)
        dataset_div_btn.grid(row=6,column=0,rowspan=3,sticky=ctk.NSEW, padx=10,pady=5)
        settings_btn.grid(row=9,column=0,rowspan=3,sticky=ctk.NSEW, padx=10,pady=5)

        return {'hp_optim': hyperparam_optim_btn, 'model_build': model_build_btn, 'dataset_div':dataset_div_btn, 'settings': settings_btn}
    
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
         
        inProgress = InProgressWindow(self.master, self.my_font, getImgPath("check_files.gif"))
        inProgress.create()
        self.ranked_features_dicts = None
    
        def checking_excel_files ():
            valid, _ = CHECK_XLS_FILES(self.train_entryVar.get(), self.test_entryVar.get())
            if not valid:
                self.ALL_LOADED_FEATURES.set("")
                self.SELECTED_FEATURES.set("")
                inProgress.destroy()
                CustomWarningBox(
                    parent=self.master, my_font=self.my_font, 
                    warnings=["Train and Test files do not have identical column sets."]
                )
                return
        
        def load_ranked_features ():
            self.ranked_features_dicts = GET_RANKED_FEATURES(self.train_entryVar.get())
            self.ALL_LOADED_FEATURES.set(json.dumps(self.ranked_features_dicts))
        
        def populate_loaded_features_to_selected ():
            MIN_COUNT_OF_FEATURES_TO_SELECT = int(_COMMON_PROPS['feature_selection']['min_features_selected'])
            selected_features = [
                rec['Feature'] for rec in self.ranked_features_dicts 
                if rec['Rank']<=MIN_COUNT_OF_FEATURES_TO_SELECT
            ]
            self.SELECTED_FEATURES.set(",".join(selected_features))

        def step_2():
            """ Step 2: Update progress & start feature extraction """
            inProgress.update_progress_verdict('XLS files accepted.\nExtracting Features (with Ranks) ..')
            Thread(target=lambda: async_wrapper(load_ranked_features, step_3)).start()

        def step_3():
            """ Step 3: Update progress & load ranked features """
            inProgress.update_progress_verdict('Loading Ranked Features..\nAlmost There!!')
            self.master.after(100, check_ranked_features_ready)  # Check when ranked features are available

        def check_ranked_features_ready():
            """ Waits until ranked_features_dicts is not None before proceeding """
            if self.ranked_features_dicts is not None:
                populate_loaded_features_to_selected()
                self.master.after(200, inProgress.destroy)
            else:
                self.master.after(100, check_ranked_features_ready)  # Retry after 100ms

        def async_wrapper(func, callback):
            """ Runs a function in a thread & calls the callback in the main thread """
            func()
            self.master.after(100, callback)  # Schedule callback on the main thread

        # Step 1: Check files in a separate thread & continue after completion
        inProgress.update_progress_verdict('Checking\nXLS files..')
        Thread(target=lambda: async_wrapper(checking_excel_files, step_2)).start()

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
    
class DataSetDivFrame:
    def __init__(self, masterFrame: ctk.CTkFrame, my_font: ctk.CTkFont, colors: dict, img_pathsAndSizes: dict):
        self.master = masterFrame
        self.my_font = my_font
        self.colors = colors
        self.img_pathsAndSizes = img_pathsAndSizes
        self.methods_of_div_dict = {
            'KS': 'Kennard Stone', 
            'RANDOM':'Random Split', 
            'ACTIVITY': 'Activity-Based'
        }
        # VARIABLES
        self.dataset_selected_var = ctk.StringVar()
        self.method_of_div_var = ctk.StringVar(value=self.methods_of_div_dict['KS'])
        self.percent_of_samples_var = ctk.IntVar(value=80)
        self.seed_number_var = ctk.IntVar(value=3)
        self.num_of_samples_per_cluster_var = ctk.IntVar(value=5)

        self.dataset_div_mainFrame = self._get_labelframe(self.master, 'Dataset_division')
        self.dataset_div_mainFrame.grid_columnconfigure(tuple(range(7)), weight=1)
        #self.dataset_div_mainFrame.grid_rowconfigure(1, weight=1) # Results
        self.dataset_div_mainFrame.grid(row=0, column=0, padx=30, pady=30, sticky=ctk.NSEW)

        # DATASET SELECT ENTRY
        self.dataset_select_label = ctk.CTkLabel(
            master=self.dataset_div_mainFrame, text='SELECT DATASET:', font=self.my_font
        )
        self.dataset_select_entry = self.create_entry(self.dataset_selected_var)
        self.dataset_select_btn = self.create_upload_button()

        self.dataset_select_label.grid(row=0, column=0, columnspan=1, padx=5, pady=5, sticky=ctk.EW)
        self.dataset_select_entry.grid(row=0, column=1, columnspan=4, padx=5, pady=5, sticky=ctk.EW)
        self.dataset_select_btn.grid(row=0, column=5, columnspan=1, padx=5, pady=5, sticky=ctk.EW)

        # METHOD OF DIV.
        self.method_of_div_label = ctk.CTkLabel(
            master=self.dataset_div_mainFrame, text="METHOD of DIV:", font=self.my_font
        )
        self.method_of_div_menu = ctk.CTkOptionMenu(
            master=self.dataset_div_mainFrame, 
            variable=self.method_of_div_var,
            values=list(self.methods_of_div_dict.values()),
            command=self.load_options,
            font=self.my_font,
            dropdown_font=self.my_font,
            corner_radius=0,
            anchor=ctk.CENTER,
            button_color=self.colors['optionmenu']['fg'],
            button_hover_color=self.colors['optionmenu']['hover'],
            fg_color=self.colors['optionmenu']['fg']    
        )

        self.method_of_div_label.grid(row=1, column=0, columnspan=1, padx=5, pady=5, sticky=ctk.EW)
        self.method_of_div_menu.grid(row=1, column=1, columnspan=6, padx=5, pady=5, sticky=ctk.EW)
        
        # DYNAMIC OPTION FRAME [BASED on METHOD_OF_DIV]
        self.method_of_div_options_frame = ctk.CTkFrame(
            master=self.dataset_div_mainFrame,
            fg_color=self.colors['fg'],
        )
        self.method_of_div_options_frame.grid_columnconfigure((0,1), weight=1)
        self.method_of_div_options_frame.grid(row=2, column=0, columnspan=7, padx=5, pady=5, sticky=ctk.EW)

        # >>> OPTIONS
        self.load_options(self.method_of_div_var.get())

        # SUBMIT
        self.submit_btn = ctk.CTkButton(
            master=self.dataset_div_mainFrame,
            text='Submit',
            font=self.my_font,
            fg_color=self.colors['btn']['fg'],
            hover_color=self.colors['btn']['hover'],
            text_color='white',
            corner_radius=0,
            width=200,
            border_spacing=0,
            command=self.submit
        )  
        self.submit_btn.grid(row=3, column=0, columnspan=7, padx=25, pady=5)

    def load_options(self, selection):
        # Clear previous widgets
        if self.method_of_div_options_frame.winfo_children():
            for widget in self.method_of_div_options_frame.winfo_children():
                widget.destroy()
        
        if selection == self.methods_of_div_dict['KS']:
            self.percent_of_samples_label = ctk.CTkLabel(
                self.method_of_div_options_frame, 
                text="% of Train Samples (51-95):",
                font=self.my_font
            )
            self.percent_of_samples_entry = ctk.CTkEntry(
                master=self.method_of_div_options_frame,
                textvariable=self.percent_of_samples_var,
                border_width=0,
                corner_radius=0,
                font=self.my_font
            )
            self.percent_of_samples_label.grid(row=0, column=0, padx=5, pady=5, sticky=ctk.E)
            self.percent_of_samples_entry.grid(row=0, column=1, padx=5, pady=5, sticky=ctk.W)
        
        elif selection == self.methods_of_div_dict['RANDOM']:
            self.seed_number_var.set(value=42)

            self.percent_of_samples_label = ctk.CTkLabel(
                self.method_of_div_options_frame, text="% of Train Samples (51-95):", font=self.my_font
            )
            self.percent_of_samples_entry = ctk.CTkEntry(
                master=self.method_of_div_options_frame,
                textvariable=self.percent_of_samples_var,
                border_width=0,
                corner_radius=0,
                font=self.my_font
            )
            self.seed_number_label = ctk.CTkLabel(
                self.method_of_div_options_frame, text="Seed Number:", font=self.my_font
            )
            self.seed_number_entry = ctk.CTkEntry(
                master=self.method_of_div_options_frame,
                textvariable=self.seed_number_var,
                border_width=0,
                corner_radius=0,
                font=self.my_font
            )

            self.percent_of_samples_label.grid(row=0, column=0, padx=5, pady=5, sticky=ctk.E)
            self.percent_of_samples_entry.grid(row=0, column=1, padx=5, pady=5, sticky=ctk.W)
            self.seed_number_label.grid(row=1, column=0, padx=5, pady=5, sticky=ctk.E)
            self.seed_number_entry.grid(row=1, column=1, padx=5, pady=5, sticky=ctk.W)
        
        elif selection == self.methods_of_div_dict['ACTIVITY']:
            self.num_of_samples_per_cluster_var.set(value=5)
            self.seed_number_var.set(value=3)

            self.seed_number_label = ctk.CTkLabel(
                self.method_of_div_options_frame, text="Seed Number :", font=self.my_font
            )
            self.seed_number_entry = ctk.CTkEntry(
                master=self.method_of_div_options_frame,
                textvariable=self.seed_number_var,
                border_width=0,
                corner_radius=0,
                font=self.my_font
            )
            self.num_of_samples_per_cluster_label = ctk.CTkLabel(
                self.method_of_div_options_frame, text="Num of Samples per Cluster :", font=self.my_font
            )
            self.num_of_samples_per_cluster_entry = ctk.CTkEntry(
                master=self.method_of_div_options_frame,
                textvariable=self.num_of_samples_per_cluster_var,
                border_width=0,
                corner_radius=0,
                font=self.my_font
            )

            self.num_of_samples_per_cluster_label.grid(row=0, column=0, padx=5, pady=5, sticky=ctk.E)
            self.num_of_samples_per_cluster_entry.grid(row=0, column=1, padx=5, pady=5, sticky=ctk.W)
            self.seed_number_label.grid(row=1, column=0, padx=5, pady=5, sticky=ctk.E)
            self.seed_number_entry.grid(row=1, column=1, padx=5, pady=5, sticky=ctk.W)

    def submit(self):
        inProgress = InProgressWindow(self.master, self.my_font, getImgPath('dataset_div_loading.gif'))
        inProgress.create()
        
        self.files_created = None
        def update_success():
            inProgress.destroy()
            success_msg = "Calculations Completed !!"
            if self.files_created and type(self.files_created)==tuple:
                success_msg+=f'\nTrain: {self.files_created[0]}'
                success_msg+=f'\nTest: {self.files_created[1]}'

            CustomSuccessBox(self.master, success_msg, self.my_font)
            
        def update_failure(warnings: list):
            inProgress.destroy()
            CustomWarningBox(self.master, warnings, self.my_font)

        if not self.dataset_selected_var.get():
            self.master.after(1000, lambda warnings=['No DATASET selected !!']: update_failure(warnings))
            return
        
        dataset_div_method_selected = self.method_of_div_var.get()
        if dataset_div_method_selected == self.methods_of_div_dict['KS']:
            try:
                percent_of_samples = int(self.percent_of_samples_var.get())
                if not 51<percent_of_samples<=95:
                    raise Exception('Train Samples % must be a in range [51,95]')
                # ...
                print('VALID')
                def run_dataset_division():
                    self.files_created = KENNARD_STONE(
                        dataset_file_path = self.dataset_selected_var.get(),
                        train_samples_percent = percent_of_samples,
                        inProgress = inProgress
                    )
                    self.master.after(1000, update_success)

                Thread(target=run_dataset_division).start()
            except Exception as ex:
                self.master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))
                return
            
        elif dataset_div_method_selected == self.methods_of_div_dict['ACTIVITY']:
            try:
                num_of_samples_per_cluster = int(self.num_of_samples_per_cluster_var.get())
                if not 0<num_of_samples_per_cluster<=100:
                    raise Exception('Num Of Samples per Cluster must be a in range [1,100]')
                
                seed_number = int(self.seed_number_var.get())
                if not 0<seed_number<=num_of_samples_per_cluster:
                    raise Exception(f'Seed Number must be a in range [1,{num_of_samples_per_cluster}]')
                
                print('VALID')
                def run_dataset_division():
                    self.files_created = ACTIVITY_BASED_DIV(
                        dataset_file_path = self.dataset_selected_var.get(),
                        num_of_compounds_in_each_cluster=num_of_samples_per_cluster,
                        seed_number=seed_number,
                        inProgress = inProgress
                    )
                    self.master.after(1000, update_success)

                Thread(target=run_dataset_division).start()
                
            except Exception as ex:
                self.master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))
                return
            
        elif dataset_div_method_selected == self.methods_of_div_dict['RANDOM']:
            try:
                percent_of_samples = int(self.percent_of_samples_var.get())
                if not 51<percent_of_samples<=95:
                    raise Exception('Train Samples % must be a in range [51,95]')
                
                seed_number = int(self.seed_number_var.get())
                if not 0<seed_number<=100:
                    raise Exception('Seed Number must be a in range [1,100]')
                # ...
                print('VALID')
                def run_dataset_division():
                    self.files_created = RANDOM_DIV(
                        dataset_file_path = self.dataset_selected_var.get(),
                        train_samples_percent=percent_of_samples,
                        seed_number=seed_number,
                        inProgress = inProgress
                    )
                    self.master.after(1000, update_success)

                Thread(target=run_dataset_division).start()
                
            except Exception as ex:
                self.master.after(1000, lambda warnings=[str(ex)]: update_failure(warnings))
                return
        else:
            self.master.after(1000, lambda warnings=[f'Invalid Method: {dataset_div_method_selected} !!']: update_failure(warnings))
            return


    def create_entry(self, entry_var):
        entry = ctk.CTkEntry(
            master=self.dataset_div_mainFrame,
            textvariable=entry_var,
            border_width=0,
            corner_radius=0,
            width=300,
            font=self.my_font
        )
        entry.bind('<FocusIn>', self.on_focus_in)
        entry.bind('<FocusOut>', self.on_focus_out)
        return entry

    def create_upload_button(self):
        upload_img = ctk.CTkImage(
            light_image=Image.open(self.img_pathsAndSizes['upload']['path']), 
            size=self.img_pathsAndSizes['upload']['size']
        )
        button = ctk.CTkButton(
            master=self.dataset_div_mainFrame,
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
            command=self.load_file
        )
        return button
    
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title=f"Select a Dataset to Divide",
            filetypes=[("Excel Files", "*.xls"), ("Excel Files", "*.xlsx")]
        )
        if file_path:
            self.dataset_selected_var.set(file_path)
    
    def on_focus_in(self):
        self.dataset_select_entry.configure(border_color="#111")

    def on_focus_out(self):
        self.dataset_select_entry.configure(border_color="#bbb")

    def _get_labelframe (self, master_frame: ctk.CTkFrame, label_txt: str):
        return tkLabelFrame(
            master=master_frame, text=label_txt, font=self.my_font, 
            labelanchor=ctk.NW, background=self.colors['fg']
        )