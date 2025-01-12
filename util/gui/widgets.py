from customtkinter import *
from PIL import Image, ImageSequence
from os import path as os_path
from data import DATA, _COMMON_PROPS

COLORS = DATA['colors']
OPTUNA_TOTAL_TRIALS = _COMMON_PROPS['hp_optim']['optuna_total_trials']

def getImgPath (img_name, image_dir='images'):
    return os_path.join(image_dir, img_name)

class FeatureSelectEntry(CTkFrame):
    def __init__(self, parent:CTkFrame, my_font:CTkFont, selectedOptionsVar:StringVar, allOptionsVar:StringVar, trainPathVar: StringVar, testPathVar: StringVar, MIN_CHOOSE:int=2):
        super().__init__(parent)
        parent.grid_columnconfigure(0, weight=1)

        self.grid(row=0, column=0)
        self.grid_columnconfigure(0, weight=1)
        self.configure(fg_color=COLORS["SKYBLUE_FG"])

        self.my_font = my_font
        self.selectedOptions_var = selectedOptionsVar
        self.allOptionsVar=allOptionsVar
        self.trainPathVar = trainPathVar
        self.testPathVar = testPathVar
        self.MIN_CHOOSE = MIN_CHOOSE
        self.search_image = CTkImage(Image.open(getImgPath("search.png")), size=(24, 24))

        self.value_label = CTkEntry(
            self, textvariable=self.selectedOptions_var ,font=my_font, border_width=0, justify=CENTER,
            fg_color="white", text_color="black", state='disabled', corner_radius=0
        )
        self.value_label.grid(row=0, column=0, sticky=NSEW)

        self.select_button = CTkButton(
            self, image=self.search_image, text="", 
            fg_color=COLORS["GREY_FG"],
            hover_color=COLORS["GREY_HOVER_FG"],
            command=self.open_featureSelectionWindow, width=50, corner_radius=0
        )
        self.select_button.grid(row=0, column=1)

    def open_featureSelectionWindow(self):
        if not self.trainPathVar.get() or not self.testPathVar.get():
            CustomWarningBox(self, ["Please select both Train and Test files first."], self.my_font)
            return
        
        from util.ml.functions import CHECK_XLS_FILES
        valid, columns = CHECK_XLS_FILES(self.trainPathVar.get(), self.testPathVar.get())
        if not valid:
            self.allOptionsVar.set("")
            self.selectedOptions_var.set("")
            CustomWarningBox(self, ["Train and Test files do not have identical column sets."], self.my_font)
        else:
            try:
                MultiSelectDialog(self, "Features", list(self.allOptionsVar.get().split(',')), self.selectedOptions_var, self.my_font, self.MIN_CHOOSE)
            except Exception as ex:
                CustomWarningBox(self, [ex], self.my_font)

class MultiSelectEntry(CTkFrame):
    def __init__(self, parent:CTkFrame, whatToChoosePlural:str, my_font:CTkFont, tkVar:StringVar, options:list[str], MIN_CHOOSE:int=2):
        super().__init__(parent)
        parent.grid_columnconfigure(0, weight=1)

        self.grid(row=0, column=0)
        self.grid_columnconfigure(0, weight=1)
        self.configure(fg_color=COLORS["SKYBLUE_FG"])

        self.whatToChoosePlural=whatToChoosePlural
        self.my_font = my_font
        self.selectedOptions_var = tkVar
        self.options=options
        self.MIN_CHOOSE = MIN_CHOOSE
        self.search_image = CTkImage(Image.open(getImgPath("search.png")), size=(24, 24))

        self.value_label = CTkEntry(
            self, textvariable=self.selectedOptions_var ,font=my_font, border_width=0, justify=CENTER,
            fg_color="white", text_color="black", state='disabled', corner_radius=0
        )
        self.value_label.grid(row=0, column=0, sticky=NSEW)

        self.select_button = CTkButton(
            self, image=self.search_image, text="", 
            fg_color=COLORS["GREY_FG"],
            hover_color=COLORS["GREY_HOVER_FG"],
            command=self.open_selection_window, width=50, corner_radius=0
        )
        self.select_button.grid(row=0, column=1)

    def open_selection_window(self):
        try:
            MultiSelectDialog(self, self.whatToChoosePlural, self.options, self.selectedOptions_var, self.my_font, self.MIN_CHOOSE)
        except Exception as ex:
            CustomWarningBox(self, [ex], self.my_font)

class MyIntegerEntry(CTkFrame):
    def __init__(self, parent:CTkFrame, my_font:CTkFont, tkVar:IntVar, min_value=1, max_value=100):
        super().__init__(parent)
        parent.grid_columnconfigure(0, weight=1)

        self.grid(row=0, column=0)
        self.grid_columnconfigure(1, weight=1)
        self.configure(fg_color=COLORS["SKYBLUE_FG"])
        
        self.min_value = min_value
        self.max_value = max_value
        self.current_value = tkVar
        self.hold_job = None  # To track repeating commands

        # Load images for buttons
        self.minus_image = CTkImage(Image.open(getImgPath("minus.png")), size=(20, 20))
        self.plus_image = CTkImage(Image.open(getImgPath("add.png")), size=(20, 20))
        
        # Decrement button with image
        self.decrement_button = CTkButton(
            self, image=self.minus_image, text="", 
            fg_color=COLORS["SKYBLUE_FG"],
            hover_color=COLORS["LIGHTRED_HOVER_FG"],
            command=self.decrement, width=20
        )
        self.decrement_button.grid(row=0, column=0)
        self.decrement_button.bind("<ButtonPress-1>", lambda event: self.start_repeat(self.decrement))
        self.decrement_button.bind("<ButtonRelease-1>", self.stop_repeat)
        
        # Value label
        self.value_label = CTkEntry(
            self, textvariable=self.current_value ,font=my_font, border_width=0, justify=CENTER,
            fg_color="white", text_color="black", state='disabled', width=50
        )
        self.value_label.grid(row=0, column=1)

        # Increment button with image
        self.increment_button = CTkButton(
            self, image=self.plus_image, text="", 
            fg_color=COLORS["SKYBLUE_FG"],
            hover_color=COLORS["MEDIUMGREEN_HOVER_FG"],
            command=self.increment, width=20
        )
        self.increment_button.grid(row=0, column=2)
        self.increment_button.bind("<ButtonPress-1>", lambda event: self.start_repeat(self.increment))
        self.increment_button.bind("<ButtonRelease-1>", self.stop_repeat)
    
    def increment(self):
        """Increase the current value, ensuring it doesn't exceed max_value."""
        if self.current_value.get() < self.max_value:
            self.current_value.set(self.current_value.get()+1)
    
    def decrement(self):
        """Decrease the current value, ensuring it doesn't go below min_value."""
        if self.current_value.get() > self.min_value:
            self.current_value.set(self.current_value.get()-1)

    def start_repeat(self, command):
        """Start repeating the given command."""
        if self.hold_job is None:
            self.hold_job = self.after(100, lambda: self.repeat_command(command))

    def repeat_command(self, command):
        """Repeat the command while the button is held."""
        command()
        self.hold_job = self.after(100, lambda: self.repeat_command(command))

    def stop_repeat(self, event):
        """Stop repeating the command when the button is released."""
        if self.hold_job is not None:
            self.after_cancel(self.hold_job)
            self.hold_job = None

class MyFloatingLogEntry(CTkFrame):
    def __init__(self, parent: CTkFrame, my_font: CTkFont, tkVar: DoubleVar, min_value=1e-5, max_value=1e5):
        super().__init__(parent)
        parent.grid_columnconfigure(0, weight=1)

        self.grid(row=0, column=0)
        self.grid_columnconfigure(1, weight=1)
        self.configure(fg_color=COLORS["SKYBLUE_FG"])

        self.min_value = min_value
        self.max_value = max_value
        self.current_value = tkVar
        self.hold_job = None  # To track repeating commands

        # Bind a trace to update the display value
        self.current_value.trace_add("write", self.update_display)

        # Load images for buttons
        self.minus_image = CTkImage(Image.open(getImgPath("minus.png")), size=(24, 24))
        self.plus_image = CTkImage(Image.open(getImgPath("add.png")), size=(24, 24))

        # Decrement button
        self.decrement_button = CTkButton(
            self, image=self.minus_image, text="",
            fg_color=COLORS["SKYBLUE_FG"],
            hover_color=COLORS["LIGHTRED_HOVER_FG"],
            command=self.decrement, width=20
        )
        self.decrement_button.grid(row=0, column=0)
        self.decrement_button.bind("<ButtonPress-1>", lambda event: self.start_repeat(self.decrement))
        self.decrement_button.bind("<ButtonRelease-1>", self.stop_repeat)

        # Value label
        self.value_label = CTkEntry(
            self, font=my_font, border_width=0, justify=CENTER,
            fg_color="white", text_color="black", state='readonly', width=80
        )
        self.value_label.grid(row=0, column=1)

        # Increment button
        self.increment_button = CTkButton(
            self, image=self.plus_image, text="",
            fg_color=COLORS["SKYBLUE_FG"],
            hover_color=COLORS["MEDIUMGREEN_HOVER_FG"],
            command=self.increment, width=20
        )
        self.increment_button.grid(row=0, column=2)
        self.increment_button.bind("<ButtonPress-1>", lambda event: self.start_repeat(self.increment))
        self.increment_button.bind("<ButtonRelease-1>", self.stop_repeat)

        # Initialize the display
        self.update_display()

    def update_display(self, *args):
        """Update the entry field with the formatted value."""
        value = self.current_value.get()
        formatted_value = f"{value:.6f}".rstrip("0").rstrip(".")
        self.value_label.configure(state="normal")  # Temporarily enable editing
        self.value_label.delete(0, "end")
        self.value_label.insert(0, formatted_value)
        self.value_label.configure(state="readonly")  # Revert to readonly

    def increment(self):
        """Multiply the current value by 10, ensuring it doesn't exceed max_value."""
        new_value = self.current_value.get() * 10
        if new_value <= self.max_value:
            self.current_value.set(new_value)

    def decrement(self):
        """Divide the current value by 10, ensuring it doesn't go below min_value."""
        new_value = self.current_value.get() / 10
        if new_value >= self.min_value:
            self.current_value.set(new_value)

    def start_repeat(self, command):
        """Start repeating the given command."""
        if self.hold_job is None:
            self.hold_job = self.after(100, lambda: self.repeat_command(command))

    def repeat_command(self, command):
        """Repeat the command while the button is held."""
        command()
        self.hold_job = self.after(100, lambda: self.repeat_command(command))

    def stop_repeat(self, event):
        """Stop repeating the command when the button is released."""
        if self.hold_job is not None:
            self.after_cancel(self.hold_job)
            self.hold_job = None

class MyRangeEntry(CTkFrame):
    def __init__(self, parent:CTkFrame,
            from_var:IntVar, to_var:IntVar,
            my_font:CTkFont, MIN_VAL, MAX_VAL, **kwargs
        ):
        super().__init__(parent, **kwargs)
        self.configure(fg_color=COLORS["SKYBLUE_FG"])

        # Assign the IntVars to the instance
        self.from_var = from_var
        self.to_var = to_var

        # Validation logic
        self.from_var.trace_add("write", self.sync_to_var)
        self.to_var.trace_add("write", self.sync_from_var)

        self.label_from = CTkLabel(self, text="From:", font=my_font, fg_color=COLORS["SKYBLUE_FG"])
        self.label_from.grid(row=0, column=0, padx=2, pady=5, sticky="e")
        self.entry_from = MyIntegerEntry(parent=self, my_font=my_font, tkVar=self.from_var, min_value=MIN_VAL, max_value=MAX_VAL-1)
        self.entry_from.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.label_to = CTkLabel(self, text="To:", font=my_font, fg_color=COLORS["SKYBLUE_FG"])
        self.label_to.grid(row=0, column=2, padx=2, pady=5, sticky="e")
        self.entry_to = MyIntegerEntry(parent=self, my_font=my_font, tkVar=self.to_var, min_value=MIN_VAL+1, max_value=MAX_VAL)
        self.entry_to.grid(row=0, column=3, padx=5, pady=5, sticky="w")
    
    def sync_to_var(self, *args):
        from_value = self.from_var.get()
        to_value = self.to_var.get()

        if from_value == to_value:
            self.to_var.set(from_value+1)

    def sync_from_var(self, *args):
        from_value = self.from_var.get()
        to_value = self.to_var.get()

        if to_value == from_value:
            self.from_var.set(to_value-1)

class MyFloatingLogRangeEntry(CTkFrame):
    def __init__(self, parent: CTkFrame,
                 from_var: DoubleVar, to_var: DoubleVar,
                 my_font: CTkFont, MIN_VAL: float, MAX_VAL: float, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(fg_color=COLORS["SKYBLUE_FG"])

        # Assign the DoubleVars to the instance
        self.from_var = from_var
        self.to_var = to_var
        self.MIN_VAL = MIN_VAL
        self.MAX_VAL = MAX_VAL

        # Validation logic
        self.from_var.trace_add("write", self.sync_to_var)
        self.to_var.trace_add("write", self.sync_from_var)

        self.label_from = CTkLabel(self, text="From:", font=my_font, fg_color=COLORS["SKYBLUE_FG"])
        self.label_from.grid(row=0, column=0, padx=2, pady=5, sticky="e")
        self.entry_from = MyFloatingLogEntry(
            parent=self, my_font=my_font, tkVar=self.from_var,
            min_value=self.MIN_VAL, max_value=self.MAX_VAL/10
        )
        self.entry_from.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.label_to = CTkLabel(self, text="To:", font=my_font, fg_color=COLORS["SKYBLUE_FG"])
        self.label_to.grid(row=0, column=2, padx=2, pady=5, sticky="e")
        self.entry_to = MyFloatingLogEntry(
            parent=self, my_font=my_font, tkVar=self.to_var,
            min_value=self.MIN_VAL*10, max_value=self.MAX_VAL
        )
        self.entry_to.grid(row=0, column=3, padx=5, pady=5, sticky="w")

    def formatted_value(self, val):
        """Format a float value for comparison."""
        return f"{val:.10f}".rstrip("0").rstrip(".")

    def sync_to_var(self, *args):
        """Ensure `to_var` stays greater than `from_var`."""
        from_value = self.from_var.get()
        to_value = self.to_var.get()

        # If to_value is less than or equal to from_value, adjust to_value
        if to_value <= from_value:
            self.to_var.set(from_value * 10)

    def sync_from_var(self, *args):
        """Ensure `from_var` stays less than `to_var`."""
        from_value = self.from_var.get()
        to_value = self.to_var.get()

        # If from_value is greater than or equal to to_value, adjust from_value
        if from_value >= to_value:
            self.from_var.set(to_value / 10)

class MyStepRangeEntry(MyRangeEntry):
    def __init__(self, parent:CTkFrame,
            from_var:IntVar, to_var:IntVar, step_var:IntVar, 
            my_font:CTkFont, MIN_VAL, MAX_VAL, MAX_STEPS, **kwargs
        ):
        super().__init__(parent, from_var, to_var, my_font, MIN_VAL, MAX_VAL, **kwargs)

        # Assign the IntVars to the instance
        self.step_var = step_var

        self.label_step = CTkLabel(self, text="Step:", font=my_font, fg_color=COLORS["SKYBLUE_FG"])
        self.label_step.grid(row=0, column=4, padx=5, pady=5, sticky="e")
        self.entry_step = MyIntegerEntry(parent=self, my_font=my_font, tkVar=self.step_var, min_value=1, max_value=MAX_STEPS)
        self.entry_step.grid(row=0, column=5, padx=5, pady=5, sticky="w")

class MyStepRangeEntry1(CTkFrame):
    def __init__(self, parent:CTkFrame,
            from_var:IntVar, to_var:IntVar, step_var:IntVar, 
            my_font:CTkFont, MIN_VAL, MAX_VAL, MAX_STEPS, **kwargs
        ):
        super().__init__(parent, **kwargs)
        self.configure(fg_color=COLORS["SKYBLUE_FG"])

        # Assign the IntVars to the instance
        self.from_var = from_var
        self.to_var = to_var
        self.step_var = step_var

        # Validation logic
        self.from_var.trace_add("write", self.sync_to_var)
        self.to_var.trace_add("write", self.sync_from_var)

        self.label_from = CTkLabel(self, text="From:", font=my_font, fg_color=COLORS["SKYBLUE_FG"])
        self.label_from.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.entry_from = MyIntegerEntry(parent=self, my_font=my_font, tkVar=self.from_var, min_value=MIN_VAL, max_value=MAX_VAL-1)
        self.entry_from.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.label_to = CTkLabel(self, text="To:", font=my_font, fg_color=COLORS["SKYBLUE_FG"])
        self.label_to.grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.entry_to = MyIntegerEntry(parent=self, my_font=my_font, tkVar=self.to_var, min_value=MIN_VAL+1, max_value=MAX_VAL)
        self.entry_to.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        self.label_step = CTkLabel(self, text="Step:", font=my_font, fg_color=COLORS["SKYBLUE_FG"])
        self.label_step.grid(row=0, column=4, padx=5, pady=5, sticky="e")
        self.entry_step = MyIntegerEntry(parent=self, my_font=my_font, tkVar=self.step_var, min_value=1, max_value=MAX_STEPS)
        self.entry_step.grid(row=0, column=5, padx=5, pady=5, sticky="w")
    
    def sync_to_var(self, *args):
        from_value = self.from_var.get()
        to_value = self.to_var.get()

        if from_value == to_value:
            self.to_var.set(from_value+1)

    def sync_from_var(self, *args):
        from_value = self.from_var.get()
        to_value = self.to_var.get()

        if to_value == from_value:
            self.from_var.set(to_value-1)

class MultiSelectDialog(CTkToplevel):
    def __init__(self, parent:CTkFrame, 
            whatToChoosePlural:str, options:list[str], selectedOptions_StringVar: StringVar, 
            my_font:CTkFont, MIN_CHOOSE: int = 2
        ):
        super().__init__(parent)
        self.title(f'Choose {whatToChoosePlural}')
        self.geometry("300x300")
        self.resizable(False, False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.MIN_CHOOSE=MIN_CHOOSE

        self.selectedOptions_StringVar = selectedOptions_StringVar
        self.selectedOptions = set(selectedOptions_StringVar.get().split(','))
        if len(self.selectedOptions)<MIN_CHOOSE:
            raise Exception(f'Num. of selectedOptions must be {self.MIN_CHOOSE} or more for {whatToChoosePlural}')

        # Row0,1,2,3: ScrollableFrame (with Label)
        scrollable_frame = CTkScrollableFrame(
            self, 
            label_text=f'{whatToChoosePlural} List', 
            label_font=my_font,
            label_fg_color=COLORS['LIGHTRED_FG'],
            label_text_color='white',
            scrollbar_button_color='#333',
            scrollbar_button_hover_color=COLORS['GREY_HOVER_FG'],
            fg_color=COLORS['SKYBLUE_FG']
        )
        scrollable_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky=NSEW)
        scrollable_frame.grid_columnconfigure(0, weight=1)

        self.checkboxes = {}
        for idx, option in enumerate(options):
            ctkCheckBox = CTkCheckBox(
                scrollable_frame,
                text=option,
                font=my_font,
                corner_radius=0,
                fg_color=COLORS['MEDIUMGREEN_FG'],
                hover_color=COLORS['MEDIUMGREEN_HOVER_FG'],
                border_width=1,
                border_color='#000',
                command=lambda opt=option: self.check_num_of_options(opt)
            )
            ctkCheckBox.grid(row=idx, column=0, padx=10, pady=(10, 0), sticky=W)
            if option in self.selectedOptions:
                ctkCheckBox.select()
            self.checkboxes[option]=ctkCheckBox

        CTkLabel(
            scrollable_frame, text=f'* Num. of {whatToChoosePlural} must be >= {self.MIN_CHOOSE} !!', text_color='red',
            font=my_font, anchor=CENTER
        ).grid(row=idx+1, column=0, padx=10, pady=(10, 0), sticky=NSEW)

        # Row4: Submit Btn
        submit_button = CTkButton(
            self, 
            text="Submit", 
            font=my_font,
            fg_color=COLORS['MEDIUMGREEN_FG'],
            hover_color=COLORS['MEDIUMGREEN_HOVER_FG'],
            text_color='white',
            corner_radius=0,
            width=100,
            border_spacing=0,
            command=self.submit
        )
        submit_button.grid(row=4, column=0, padx=10, pady=10, sticky=EW)

    def submit(self):
        # Update selected options based on the state of the checkboxes
        self.selectedOptions=[]
        for option, checkBox in self.checkboxes.items():
            if checkBox.get():
                self.selectedOptions.append(option)

        # Update the StringVar with the selected options
        self.selectedOptions_StringVar.set(
            ','.join(sorted(self.selectedOptions)).strip(',')
        )
        self.destroy()

    def check_num_of_options(self, choice):
        # If choice is DESELECTED and Num.of selected < MIN_CHOOSE
        if self.checkboxes[choice].get()==0 and sum(cbox.get() for cbox in self.checkboxes.values()) < self.MIN_CHOOSE:
            self.checkboxes[choice].select()
    
class SyncableTextBox(CTkTextbox):
    def __init__(self, master, text_variable: StringVar, my_font: CTkFont, **kwargs):
        """
        A CTkTextbox synchronized with a StringVar.
        :param master: The parent widget.
        :param text_variable: A StringVar to synchronize with the textbox content.
        """
        super().__init__(master, font=my_font, fg_color=COLORS["LIGHT_YELLOW_FG"] ,**kwargs)
        self.text_variable = text_variable

        # Bind the StringVar to update the textbox when it changes
        self.text_variable.trace_add("write", self._update_textbox)

        # Insert initial content from StringVar
        self.insert("1.0", self.text_variable.get())

        # Set "normal" for optionally enable editing, Set to "disabled" for read-only
        self.configure(state="disabled")  

        # Bind to update the StringVar when the content of the textbox changes
        self.bind("<KeyRelease>", self._update_stringvar)

    def _update_textbox(self, *args):
        """Update the content of the textbox when the StringVar changes."""
        self.configure(state="normal")  # Temporarily enable editing to update content
        self.delete("1.0", "end")
        self.insert("1.0", self.text_variable.get())
        self.configure(state="disabled")  # Keep non-editable (or set to "disabled" for read-only)

    def _update_stringvar(self, event=None):
        """Update the StringVar when the content of the textbox changes."""
        self.text_variable.set(self.get("1.0", "end-1c"))

class InProgressWindow:
    def __init__(self, parent, font: str, gif_path: str):
        self.parent = parent
        self.progress_window = None
        self.gif_label = None
        self.gif_path = gif_path
        self.my_font= font
        self.created_optuna_progress=False
    
    def GET_PROGRESS_WINDOW(self):
        ''' ToBeUsed when progress_window needs to dynamically updated during PROCESS '''
        if self.is_active():
            return self.progress_window
        raise Exception('Progress Window is Not Found !!!')

    def _CREATE_OPTUNA_PROGRESS (self):
        print('[_CREATE_OPTUNA_PROGRESS]: ', self.is_active(), '; ',self.created_optuna_progress)
        if self.is_active() and not self.created_optuna_progress:
            print("CREATING..")
            self.progress_window.geometry("300x300")

            # Add labels and progress bar
            self.progress_label = CTkLabel(self.progress_window, text="Running Optimization...", font=self.my_font)
            self.progress_label.pack(pady=2,padx=10)

            self.progress_bar = CTkProgressBar(
                self.progress_window, orientation="horizontal", 
                width=250, height=5, progress_color='#0dff00', fg_color='red', border_width=0
            )
            self.progress_bar.pack(pady=5,padx=10)
            self.progress_bar.set(0)  # Initialize progress to 0

            self.trial_status_label = CTkLabel(self.progress_window, text="Trial 0/0", font=self.my_font)
            self.trial_status_label.pack(pady=1,padx=10)

            self.best_trial_label = CTkLabel(self.progress_window, text="Best Score: N/A", font=self.my_font)
            self.best_trial_label.pack(pady=10,padx=10)

            self.created_optuna_progress=True

    # Callback function to update the progress bar and text
    def _UPDATE_OPTUNA_PROGRESS_BAR (self, study, trial):
        print('[_UPDATE_OPTUNA_PROGRESS_BAR]')
        if not self.created_optuna_progress:
            raise Exception('Cannot update Optuna Progress as it is not created !!')
        
        print('[_UPDATE_OPTUNA_PROGRESS_BAR]...')
        progress = len(study.trials) / OPTUNA_TOTAL_TRIALS
        self.progress_bar.set(progress)
        self.trial_status_label.configure(text=f"Trial {len(study.trials)}/{OPTUNA_TOTAL_TRIALS}")
        
        # Update best trial info
        if study.best_trial:
            best_score = study.best_value
            self.best_trial_label.configure(text=f"Best Score: {best_score:.4f} (Trial {study.best_trial.number})")
        self.parent.update_idletasks()

    def _COMPLETE_OPTUNA_PROGRESS (self):
        self.progress_label.configure(text="Optimization Complete!")
        self.progress_bar.set(1)

    def create(self):
        # Only create the progress window if it doesn't already exist
        if not self.is_active():
            self.progress_window = CTkToplevel(self.parent, fg_color='white')  # Directly reference root
            self.progress_window.geometry("200x200")
            self.progress_window.title("Fetching..")

            # Label to display the GIF
            self.gif_label = CTkLabel(
                self.progress_window, 
                text="Fetching Results ...", 
                compound=TOP,
                text_color=COLORS["MEDIUMGREEN_FG"],
                font=self.my_font
            )
            self.gif_label.pack(pady=20)

            # Load and play GIF using CTkImage
            gif_image = Image.open(self.gif_path)
            frames = [CTkImage(frame.copy(), size=(100, 100)) for frame in ImageSequence.Iterator(gif_image)]

            def play_gif(frame=0):
                self.gif_label.configure(image=frames[frame])
                frame = (frame + 1) % len(frames)  # Loop the GIF
                self.progress_window.after(100, lambda: play_gif(frame))
            play_gif()  # Start the GIF animation

    def destroy(self):
        # Only destroy the progress window if it exists
        if self.is_active():
            self.progress_window.destroy()

    def is_active(self):
        # Check if the progress window is currently displayed
        return self.progress_window and self.progress_window.winfo_exists()
    
class CustomWarningBox:
    def __init__(self, parent, warnings, my_font):
        self.parent = parent
        self.warnings = warnings
        self.my_font = my_font

        self.warning_box = CTkToplevel(parent, fg_color='white')
        self.warning_box.title("Warnings")

        # Configure grid to center elements
        self.warning_box.grid_rowconfigure(0, weight=1)  # Space above the image
        self.warning_box.grid_rowconfigure(1, weight=1)  # Space for warnings
        self.warning_box.grid_rowconfigure(2, weight=1)  # Space for button
        self.warning_box.grid_columnconfigure(0, weight=1)  # Center all columns

        self.create_widgets()

    def create_widgets(self):
        # Add an icon or image at the top
        warningImg = CTkImage(Image.open(getImgPath('warning1.png')), size=(100, 100))

        img_label = CTkLabel(
            master=self.warning_box, 
            image=warningImg, 
            text="Warning: Please check the issues below!",
            compound=TOP,
            bg_color="white",
            text_color="red",
            font=self.my_font,
            anchor=CENTER
        )
        img_label.grid(row=0, column=0, pady=10, padx=10)

        # Add the warnings list with bullet points
        warnings_frame = CTkFrame(self.warning_box, fg_color='white')
        warnings_frame.grid(row=1, column=0, padx=20)

        for idx, warning in enumerate(self.warnings, start=1):
            warning_label = CTkLabel(
                master=warnings_frame,
                text=f"{idx}. {warning}" if len(self.warnings)>1 else f"{warning}",
                bg_color="white",
                font=self.my_font,
                wraplength=400
            )
            warning_label.grid(row=idx, column=0, sticky=W, padx=20, pady=5, columnspan=2)

        # Add a Close button
        close_button = CTkButton(
            master=self.warning_box, 
            text="OK",
            fg_color=COLORS['MEDIUMGREEN_FG'],
            hover_color=COLORS['MEDIUMGREEN_HOVER_FG'],
            text_color='white',
            border_spacing=0, 
            corner_radius=0,
            font=self.my_font,
            command=self.warning_box.destroy
        )
        close_button.grid(row=2, column=0, pady=10)

        # Keep the window on top and modal
        self.warning_box.transient(self.parent)
        self.warning_box.grab_set()
        self.warning_box.resizable(False, False)

class CustomSuccessBox:
    def __init__(self, parent, message, my_font):
        self.parent = parent
        self.message = message
        self.my_font = my_font

        self.message_box = CTkToplevel(parent, fg_color='white')
        self.message_box.title("Success")

        # Configure grid to center elements
        self.message_box.grid_rowconfigure(0, weight=1)  # Space above the image
        self.message_box.grid_rowconfigure(1, weight=1)  # Space for messages
        self.message_box.grid_rowconfigure(2, weight=1)  # Space for button
        self.message_box.grid_columnconfigure(0, weight=1)  # Center all columns

        self.create_widgets()

    def create_widgets(self):
        # Add an icon or image at the top
        successImg = CTkImage(Image.open(getImgPath('success1.png')), size=(100, 100))

        img_label = CTkLabel(
            master=self.message_box, 
            image=successImg, 
            text="Success",
            compound=TOP,
            bg_color="white",
            text_color="teal",
            font=self.my_font,
            anchor=CENTER
        )
        img_label.grid(row=0, column=0, pady=10, padx=10)

        # Add the messages list with bullet points
        messages_frame = CTkFrame(self.message_box, fg_color='white')
        messages_frame.grid(row=1, column=0, padx=20)

        message_label = CTkLabel(
            master=messages_frame,
            text=f"{self.message}",
            bg_color="white",
            font=self.my_font,
            wraplength=400
        )
        message_label.grid(row=1, column=0, sticky=W, padx=20, pady=5, columnspan=2)
        
        # Add a Close button
        close_button = CTkButton(
            master=self.message_box, 
            text="OK",
            fg_color=COLORS['MEDIUMGREEN_FG'],
            hover_color=COLORS['MEDIUMGREEN_HOVER_FG'],
            text_color='white',
            border_spacing=0, 
            corner_radius=0,
            font=self.my_font,
            command=self.message_box.destroy
        )
        close_button.grid(row=2, column=0, pady=10)

        # Keep the window on top and modal
        self.message_box.transient(self.parent)
        self.message_box.grab_set()
        self.message_box.resizable(False, False)