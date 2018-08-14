import time
import os
import json
import itertools
from datetime import datetime
from random import randint

import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames
from tkinter import X, Y, TOP, BOTTOM, LEFT, RIGHT, HORIZONTAL, VERTICAL, CENTER, END, BOTH, ACTIVE, WORD, N, E, NSEW, EW
import tkinter.ttk as ttk
# import Pmw
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

# import pyside2

from typing import Union, List, cast  # ,Union, override, get_type_hints
from typing import Optional as Opt
# , Num, simpTypes, simpList, compList, compDict,
from dml_thread.types import simpDict,  pathType, tkEvent

from dml_thread import somecython, mung, plot
from dmltk import panels
import dmltk

from concurrent.futures import ProcessPoolExecutor  # . ThreadPoolExecutor
from multiprocessing import cpu_count
# print(cpu_count())  # laptop = 8


def thread_open_mung_save(data_dir: pathType, output_dir: pathType = None, settings: Opt[simpDict] = None) -> None:
    """"""
    print(output_dir)
    mung_time = time.perf_counter()
    num_files = len([fily for fily in os.listdir(data_dir)
                     if fily.endswith((".txt", ".csv", ".tsv"))])

    if num_files > 0:
        print(f"{num_files} found in {data_dir}. Munging data  ...")
        if not output_dir:
            output_dir = f"{data_dir}/output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        workers = num_files if (0 < num_files < cpu_count()) else (
            cpu_count() - 1)  # eg = 5 if 5 files, but 8 if 10 files
        with ProcessPoolExecutor(max_workers=workers) as executor:
            filelist = [filename for filename in os.listdir(data_dir)]
            executor.map(mung.open_mung_save, itertools.repeat(data_dir, len(
                filelist)), filelist, itertools.repeat(output_dir), itertools.repeat(settings))
            # for filename in os.listdir(data_dir):
            # future = executor.submit(open_mung_save, data_dir, filename, output_dir)

        finish_time = time.perf_counter() - mung_time
        print(
            f"\nMunging done @ {finish_time:.2f} s \n{num_files} files @ {finish_time/num_files:.2f} s each")
    else:
        print("No data files found in directory")


class Application(ttk.Frame):
    """"""

    def __init__(self, master: tk.Tk = None) -> None:
        super().__init__(master)

        self.data_dir: pathType = os.getcwd() + R'\data'
        self.out_dir: pathType = self.data_dir + R'\output'
        self.default_settings = plot.get_json_settings(R'\settings\default_settings.json')
        self.settings_file = R'.\settings\settings.json'
        self.settings: simpDict = plot.get_json_settings(self.settings_file)

        if master is not None: 
            self.master = master

        self.master.title('Dml Mung')
        self._create_widgets()
        self.pack(expand=Y, fill=BOTH)

        style = ttk.Style()
        style.theme_create("MyStyle", parent="alt", settings={
            # gaps L, T, R, B
            "TNotebook": {"configure": {"tabmargins": [5, 5, 5, 0]}},
            "TNotebook.Tab": {"configure": {"padding": [10, 5]}, }})  # tab size [x, y]
        # style.theme_use("MyStyle")
        #style.configure("Green.TLabel", foreground="white", background="green")

        """when initated, try open a json to get the default and output dirs.
        if not found, use hardcoded defaults/os.getcwd().
        Make button so can change default dir and output dir and save the json. 

        button save matplotlib settings (also overwrite settings.json)
        Make input (radio?) for potentiostat types: Autolab, Zahner, utgard smallones, other(contact me)
        make radio button to select files or folders
        Make exe icon / tk.root icon"""

        """
        quitButton = ttk.Button(self.master,
                                      text='Quit',
                                     command=self.quit, #style="Green.TLabel")
        ) 
        quitButton.grid(row=6, column=3, columnspan=1, ipady=10, ipadx=10) 
        """

    def _create_widgets(self) -> None:
        panels.SeeDismissPanel(self)
        self._create_notebook_panel()

    def _create_notebook_panel(self) -> None:
        MainPanel = tk.Frame(self, name='demo')
        MainPanel.pack(side=TOP, fill=BOTH, expand=Y)

        # create the notebook
        nb = ttk.Notebook(MainPanel, name='notebook')
        # extend bindings to top level window (CTRL+TAB, SHIFT+CTRL+TAB, ALT+K )
        nb.enable_traversal()
        nb.pack(fill=BOTH, expand=Y, padx=2, pady=3)

        # create tabs in notebook
        self._create_home_tab(nb)
        self._create_file_tab(nb)
        self._create_batch_tab(nb)
        self._create_settings_tab(nb)
        # self._create_disabled_tab(nb)
        self._create_text_tab(nb)
        self._create_misc_tab(nb)

    def _create_home_tab(self, nb: ttk.Notebook) -> None:
        """widgets to be displayed on 'Home'"""
        frame = ttk.Frame(nb)
        nb.add(frame, text='   Home   ', underline=4)

    def _create_file_tab(self, nb: ttk.Notebook) -> None:
        """widgets to be displayed on 'File'"""
        frame = ttk.Frame(nb)
        nb.add(frame, text='   File   ', underline=3)

        data_dirButton = ttk.Button(frame,
                                    text='Select Data file',
                                    command=self.get_datapath)
        data_dirButton.grid(row=3, column=3)

        goButton = ttk.Button(frame,
                              text='Go',
                              command=lambda: thread_open_mung_save(self.data_dir, self.out_dir, self.settings))
        goButton.grid(row=4, column=3)

    def _create_batch_tab(self, nb: ttk.Notebook) -> None:
        """widgets to be displayed on 'Batch'"""
        frame = ttk.Frame(nb)
        nb.add(frame, text='  Batch  ', underline=2)

        data_dirButton = ttk.Button(frame,
                                    text='Select Data Folder',
                                    command=self.get_datapath)
        data_dirButton.grid(row=3, column=3)

        goButton = ttk.Button(frame,
                              text='Go',
                              command=lambda: thread_open_mung_save(self.data_dir, self.out_dir, self.settings))
        goButton.grid(row=4, column=3)

        tButton = ttk.Button(frame,
                             text='Test',
                             command=lambda: print(self.settings['size']))
        tButton.grid(row=6, column=3)

    def _create_settings_tab(self, nb: ttk.Notebook) -> None:
        """widgets to be displayed on 'Settings'"""
        settings_frame = ttk.Frame(nb, name='settings')
        settings_frame.rowconfigure(1, weight=1)
        settings_frame.columnconfigure((0, 1), weight=1, uniform=1)
        nb.add(settings_frame, text='Settings', underline=0,
               padding=2)  # underline = shortcut char index
        #settings_toolip = Pmw.Balloon(settings_frame)
        # Row for Msg
        msg = ["Please change settings and click save. To return to default settings, click reset. \nSettings will be applied to File and Batch plots"]   #
        settingslbl = ttk.Label(settings_frame, 
                                # wraplength='4i', 
                                justify=CENTER, anchor=N, 
                                text=''.join(msg))
        settingslbl.grid(row=0, column=0, columnspan=5, sticky='new', pady=(10, 10), padx=(10, 10))


        # Rows/cols for settings

        self.fontvar = tk.StringVar(settings_frame, 
                                    value=f'Font: {self.settings["family"]}')
        fontlbl = ttk.Label(settings_frame, textvariable=self.fontvar)

        fontsize_options = [int(x) for x in range(50)]
        self.fontsizevar = tk.IntVar(settings_frame, value=12)
        fontsizelbl = ttk.Label(settings_frame, text='Font-size: ')
        fontsizemenu = ttk.Combobox(
            settings_frame, textvariable=self.fontsizevar, values=fontsize_options, height=16, width=6, justify=RIGHT, validatecommand=self.update_settings)
        fontsizemenu.bind("<<ComboboxSelected>>", self.update_settings)

        fontcolorlbl = ttk.Label(settings_frame, 
                            text=f'Font-size:  {self.settings["color"]}')

        valign_options = ['center', 'top', 'bottom', 'baseline']
        self.fontvalignvar = tk.StringVar(settings_frame)
        fontvalignlbl = ttk.Label(settings_frame, text='Axes font colour: ')
        fontvalignmenu = ttk.OptionMenu(settings_frame, self.fontvalignvar, *valign_options)

        halign_options = ['center', 'right', 'left']
        self.fonthalignvar = tk.StringVar(settings_frame)
        fonthalignlbl = ttk.Label(settings_frame, text='Axes font colour: ')
        fonthalignmenu = ttk.OptionMenu( settings_frame, self.fonthalignvar, *halign_options)

        """    "style or fontstyle": "normal", | 'italic' | 'oblique']
        """

        fontlbl.grid(row=1, column=1, pady=(0, 0), padx=(0, 0))
        fontcolorlbl.grid(row=1, column=2, pady=(0, 0), padx=(0, 0))

        fontsizelbl.grid(row=2, column=1, pady=(0, 0), padx=(0, 0))
        fontsizemenu.grid(row=2, column=2, pady=(0, 0), padx=(0, 0))

        fontvalignlbl.grid(row=3, column=1, pady=(0, 0), padx=(0, 0))
        fontvalignmenu.grid(row=3, column=2, pady=(0, 0), padx=(0, 0))

        fonthalignlbl.grid(row=4, column=1, pady=(0, 0), padx=(0, 0))
        fonthalignmenu.grid(row=4, column=2, pady=(0, 0), padx=(0, 0))

        # Row for load 
        loadlbl = ttk.Label(settings_frame, text='Load Settings from: ')
        loadtxt = tk.Entry(settings_frame, textvariable='', width=40, justify=LEFT)
        loadtxt.insert(0, self.settings_file)
        loadbrowsebtn = ttk.Button(settings_frame,
                                   text='Browse', command=lambda: self.populate_settingspath(loadtxt))
        loadbtn = ttk.Button(settings_frame,  # populate text box and update self.settings too
                                    text='Load Settings', underline=0, command=self.load_settings)
        
        loadlbl.grid(row=8, column=1, pady=(0, 10), padx=(0, 0))  #(T, B)  (L, R)
        loadtxt.grid(row=8, column=2, pady=(0, 10), padx=(0, 10))
        loadbrowsebtn.grid(row=8, column=3, pady=(0, 10), padx=(0, 10))
        loadbtn.grid(row=8, column=4, pady=(0, 10), padx = (0, 10))
        #settings_toolip.bind(loadbtn, 'Your name \nEnter your name')
        #loadbtn.bind("<Enter>", self.load_enter)
        # loadbtn.bind("<Leave>", self.load_leave)

        # Row for save
        savelbl = ttk.Label(settings_frame, text='Save Settings to: ')
        savetxt = tk.Entry(settings_frame, textvariable='',
                           width=40, justify=LEFT)
        savetxt.insert(0, f'{self.settings_file[:-5]}_{datetime.now().year}_{randint(0, 1000)}.json')
        savebrowsebtn = ttk.Button(settings_frame,
                                   text='Browse', command=lambda: self.populate_settingspath(loadtxt))

        savebtn = ttk.Button(settings_frame,  # should create json from textboxes, then save self.setting
                             text='Save Settings', underline=0, command=self.set_settingsfile)
        savelbl.grid(row=9, column=1, pady=(0, 10), padx=(0, 0))  #(T, B)  (L, R)
        savetxt.grid(row=9, column=2, pady=(0, 10), padx=(0, 10))
        savebrowsebtn.grid(row=3, column=3, pady=(0, 10), padx=(0, 10))
        savebtn.grid(row=9, column=4, pady=(0, 10), padx = (0, 10))
        #settings_toolip.bind(savebtn, 'Saves the current settings to a file. \nRandom name generated, change if desired.')

        # Row for reset
        resetbtn = ttk.Button(settings_frame, 
                                text='Reset', underline=0, command=self.reset_defaultsettingsfile)
        resetbtn.grid(row=5, column=4, pady=(0, 10), padx=(0, 10))


    def _create_misc_tab(self, nb: ttk.Notebook) -> None:
        """widgets to be displayed on 'Misc'"""
        frame = ttk.Frame(nb)
        nb.add(frame, text='Misc', underline=0)

        some_mathButton = ttk.Button(frame,
                                     text='Random Math',
                                     command=self.print_somecython)
        some_mathButton.grid(row=2, column=3)

    def _create_disabled_tab(self, nb: ttk.Notebook) -> None:
        """Populate the second pane. Note that the content doesn't really matter"""
        frame = ttk.Frame(nb)
        nb.add(frame, text='Disabled', state='disabled')

    def _create_text_tab(self, nb: ttk.Notebook) -> None:
        """populate the third frame with a text widget"""
        frame = ttk.Frame(nb)
        txt = tk.Text(frame, wrap=WORD, width=40, height=10)
        vscroll = ttk.Scrollbar(frame, orient=VERTICAL, command=txt.yview)
        txt['yscroll'] = vscroll.set
        vscroll.pack(side=RIGHT, fill=Y)
        txt.pack(fill=BOTH, expand=Y)
        # add to notebook (underline = index for short-cut character)
        nb.add(frame, text='Text Editor', underline=0)

    @staticmethod
    def print_somecython() -> None:
        a_num: Union[str, int] = ""
        while not isinstance(a_num, int):
            a_num = input("input number, press enter ")
            try:
                a_num = int(a_num)
            except:
                continue
            else:
                b_num: int = a_num
                print(somecython.somemath(b_num))

    def get_datapath(self) -> None:
        # new_data_dir: pathType = askdirectory(title='Select data folder')
        data_dir = os.getcwd() + R'\data'
        filetypes = [('All files', '*.*'), ('CSV files',
                                            '*.csv'), ('Text files', 'txt.*'), ]
        new_data_list: List[pathType] = askopenfilenames(
            title='Select data folder', filetypes=filetypes, initialdir=data_dir)
        if new_data_list:
            new_data_file: pathType = new_data_list[0]
            # directory = os.path.dirname(os.path.realpath(tkFileDialog.askopenfilename()))
            new_data_dir: pathType = cast(
                pathType, new_data_file.rsplit('/', 1)[0])
            self.data_dir = new_data_dir
            print(f'{new_data_dir} selected')
        else:
            print("No folder selected")

    def get_outputpath(self) -> None:
        new_output_dir: pathType = askdirectory(
            title='Select an output directory', initialdir=os.getcwd())
        if new_output_dir:
            print(new_output_dir)
            self.out_dir = new_output_dir

    """files = filedialog.askopenfilenames(parent=self.master,title='Choose
        the Music file(s)', filetypes=(("xml files", "*.xml")))"""

    def load_enter(self, event: tkEvent) -> None:
        print('Hovering over load...')

    def get_settingsfile(self) -> pathType:
        settings_dir = os.getcwd() + '\settings'
        # print(settings_dir)
        settings_file: pathType = askopenfilename(
            title='Select a settings file (.json)', initialdir=settings_dir)
        if settings_file:
            print(f'Settings file loaded: {settings_file}')
            self.settings_file = settings_file
        return self.settings_file

    def populate_settingspath(self, entry: tk.Entry) -> None:
        settings_path = self.get_settingsfile()
        entry.delete(0, END)
        entry.insert(0, settings_path)
        # v = tk.StringVar()
        # entry.textvariable=v
        # v.set(text)
        # s = v.get()

    def load_settings(self) -> None:
        # print(self.settings['color'])
        with open(self.settings_file, 'r') as f:
            settings_json: simpDict = dict(json.load(f))
            # reveal_type(settings_json)
            self.settings = settings_json
            default_font = matplotlib.font_manager.FontProperties()
            families = ['serif', 'sans-serif',
                        'cursive', 'fantasy', 'monospace']
            self.fontvar.set(self.settings['color'])
            self.fontvar.set(self.settings['color'])
            self.fontvar.set(self.settings['color'])
            self.fontsizevar.set(self.settings['size'])
            # print(self.settings['color'])

    def update_settings(self, event: tkEvent) -> None:
        self.settings['size'] = self.fontsizevar.get()
        print(self.settings['size'])

    def set_settingsfile(self) -> None:
        """Get text for savepath with browse btn,
        ask for filename, save, update settings_file if success"""
        settings_json = json.dumps(self.settings)
        with open(self.settings_file, 'w') as f:
            f.write(settings_json)

    def reset_defaultsettingsfile(self) -> None:
        # self.default_settings_json = plot.get_json_settings('default_settings.json')
        self.settings = self.default_settings

        print(
            f'Graphical settings reset to default (colour: {self.settings["color"]}, size: {self.settings["size"]})')
