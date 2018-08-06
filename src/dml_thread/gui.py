import time
import os
import json
import itertools

import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames
from tkinter import X, Y, TOP, BOTTOM, LEFT, RIGHT, HORIZONTAL, VERTICAL, END, BOTH, ACTIVE, WORD, N, E, NSEW, EW

import tkinter.ttk as ttk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

from typing import Union, List, cast  # ,Union, override, get_type_hints
from typing import Optional as Opt
from dml_thread.types import simpDict,  pathType  # , Num, simpTypes, simpList, compList, compDict,

from dml_thread import somecython, mung, plot
from dmltk import panels

from concurrent.futures import ProcessPoolExecutor  # . ThreadPoolExecutor
from multiprocessing import cpu_count
# print(cpu_count())  # laptop = 8 

def open_mung_save(data_dir: pathType, filename: pathType, output_dir: Opt[pathType] = None) -> None:
    """"""
    cv_file = mung.open_file_numpy(data_dir, filename)
    if cv_file is not None:
        data = mung.get_data_numpy(cv_file)
        params = mung.get_params(filename)
        if data is not None and len(data) == 2: 
            (x_var, y_var) = data
            plot.make_cv_plot(x_var, y_var, params, output_dir)
        elif data is not None and len(data) == 5:
            mung.write_imp_data(data, params, output_dir)
            mung.write_zview_data(data, params, output_dir)
            (freq_log, imped_log, phase, imag_imped, real_imped) = data
            plot.make_bode_plot(freq_log, imped_log, phase, params, output_dir)
            plot.make_nyquist_plot(imag_imped, real_imped, params, output_dir)


def thread_open_mung_save(data_dir: pathType, output_dir: pathType = None) -> None:
    """"""
    mung_time = time.perf_counter()
    num_files = len([fily for fily in os.listdir(data_dir)
                     if fily.endswith((".txt", ".csv", ".tsv"))])

    if num_files > 0:
        print(f"{num_files} found in {data_dir}. Munging data  ...")
        if not output_dir:
            output_dir = f"{data_dir}/output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        workers = num_files if (0 < num_files < cpu_count()) else (cpu_count() - 1)  # eg = 5 if 5 files, but 8 if 10 files
        with ProcessPoolExecutor(max_workers=workers) as executor:
            filelist = [filename for filename in os.listdir(data_dir)]
            executor.map(open_mung_save, itertools.repeat(data_dir, len(filelist)), filelist, itertools.repeat(output_dir))
            # for filename in os.listdir(data_dir):
            # future = executor.submit(open_mung_save, data_dir, filename, output_dir)

        finish_time = time.perf_counter() - mung_time
        print(f"\nMunging done @ {finish_time:.2f} s \n{num_files} files @ {finish_time/num_files:.2f} s each")
    else:
        print("No data files found in directory")


class Application(ttk.Frame):
    """"""
    def __init__(self, master: tk.Tk = None) -> None:
        super().__init__(master)

        self.data_dir: pathType = os.getcwd()
        self.out_dir: pathType = self.data_dir + '/output'
        self.default_settings = plot.get_json_settings('settings.json')
        self.settings_file = 'settings.json'
        self.settings = plot.get_json_settings(self.settings_file)

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
        #style.theme_use("MyStyle")
        #style.configure("Green.TLabel", foreground="white", background="green")

        """when initated, try open a json to get the default and output dirs. 
        if not found, use hardcoded defaults/os.getcwd().
        Make button so can change default dir and output dir and save the json. 
        Also button to reset to hardcoded defaults
        Same for matplotlib settings
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
                              command=lambda: thread_open_mung_save(self.data_dir))
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
                              command=lambda: thread_open_mung_save(self.data_dir))
        goButton.grid(row=4, column=3)


    def _create_settings_tab(self, nb: ttk.Notebook) -> None:
        """widgets to be displayed on 'Settings'"""
        settings_frame = ttk.Frame(nb, name='settings')
        settings_frame.rowconfigure(1, weight=1)
        settings_frame.columnconfigure((0, 1), weight=1, uniform=1)
        nb.add(settings_frame, text='Settings', underline=0,
               padding=2)  # underline = shortcut char index

        msg = ["Please change settings and click save. To return to default settings, click reset. \nSettings will be applied to File and Batch plots"]   #
        settingslbl = ttk.Label(
            settings_frame, wraplength='4i', justify=LEFT, anchor=N, text=''.join(msg))
        settingslbl.grid(row=0, column=0, columnspan=2, sticky='new', pady=5)

        loadbtn = ttk.Button(
            settings_frame, text='Load Settings', underline=0, command=None)
        loadbtn.grid(row=3, column=3, pady=(2, 4))

        savebtn = ttk.Button(
            settings_frame, text='Save Settings', underline=0, command=None)
        savebtn.grid(row=4, column=3, pady=(2, 4))

        resetbtn = ttk.Button(settings_frame, text='Reset',
                              underline=0, command=None)
        resetbtn.grid(row=5, column=3, pady=(2, 4))


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

    def print_somecython(self) -> None:
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
        filetypes = [('All files', '*.*'), ('CSV files',
                                            '*.csv'), ('Text files', 'txt.*'), ]
        new_data_list: List[pathType] = askopenfilenames(
            title='Select data folder', filetypes=filetypes)
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
        new_output_dir: pathType = askdirectory()
        if new_output_dir:
            print(new_output_dir)
            self.out_dir = new_output_dir

    """files = filedialog.askopenfilenames(parent=self.master,title='Choose
        the Music file(s)', filetypes=(("xml files", "*.xml")))"""

    def get_settingsfile(self) -> None:
        settings_file: pathType = askopenfilename()
        if settings_file:
            print(settings_file)
            self.settings_file = settings_file

    def update_settings(self) -> None:
        with open(self.settings_file, 'r') as f:
            settings_json: simpDict = json.load(f)
            self.settings = settings_json

    def set_settingsfile(self) -> None:
        settings_json = json.dumps(self.settings)
        with open(self.settings_file, 'w') as f:
            f.write(settings_json)
