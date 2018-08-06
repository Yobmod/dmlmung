import time
import os
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiofiles

import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames
import tkinter.ttk as ttk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

from typing import List, cast# ,Union, override, get_type_hints
# from typing import Optional as Opt
from dmlechemmods.types import simpDict,  pathType  # , Num, simpTypes, simpList, compList, compDict,

from dml_async import somecython, mung, plot


async def get_filelist(data_dir: pathType):
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=10)
    fut = loop.run_in_executor(executor, os.listdir, data_dir)
    yield fut


async def open_mung_save(data_dir: pathType, output_dir: pathType = None) -> None:

    mung_time = time.perf_counter()
    num_files = len([fily for fily in os.listdir(data_dir)
                     if fily.endswith((".txt", ".csv", ".tsv"))])
    print(f"{num_files} found in {data_dir}. Munging data  ....")

    if not output_dir:
        output_dir = f"{data_dir}/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    async for filename in get_filelist(data_dir):
        print(filename)
        cv_file = await mung.async_open_file_numpy(data_dir, filename)
        if cv_file is not None:
            data = await mung.get_data_numpy(cv_file)
            params = await mung.get_params(filename)
            if data is not None and len(data) == 2:
                (x_var, y_var) = data
                plot.make_cv_plot(x_var, y_var, params, output_dir)
            elif data is not None and len(data) == 5:
                await mung.write_imp_data(data, params, output_dir)
                await mung.write_zview_data(data, params, output_dir)
                (freq_log, imped_log, phase, imag_imped, real_imped) = data
                plot.make_bode_plot(freq_log, imped_log, phase, params, output_dir)
                plot.make_nyquist_plot(imag_imped, real_imped, params, output_dir)

    finish_time = round(time.perf_counter() - mung_time)
    if num_files > 0:
        print(
            f"Munging done @ {finish_time:.2f} s \n{num_files} files @ {finish_time/num_files:.2f} s each")


class Application(ttk.Frame):
    def __init__(self, master: tk.Tk = None) -> None:
        super().__init__(master)

        self.data_dir: pathType = os.getcwd()
        self.out_dir: pathType = self.data_dir + '/output'
        self.default_settings = plot.get_json_settings('settings.json')
        self.settings_file = 'settings.json'
        self.settings = plot.get_json_settings(self.settings_file)

        if master is not None:
            self.master = master
        self.master.grid_rowconfigure(0, weight=10)
        self.master.grid_rowconfigure(1, weight=10)
        self.master.grid_rowconfigure(2, weight=10)
        self.master.grid_rowconfigure(3, weight=10)
        self.master.grid_rowconfigure(4, weight=10)
        self.master.grid_rowconfigure(5, weight=10)
        self.master.grid_rowconfigure(6, weight=10)

        self.master.grid_columnconfigure(0, weight=10)
        self.master.grid_columnconfigure(1, weight=10)
        self.master.grid_columnconfigure(2, weight=10)
        self.master.grid_columnconfigure(3, weight=10)
        self.master.grid_columnconfigure(4, weight=10)
        self.master.grid_columnconfigure(5, weight=10)
        self.master.grid_columnconfigure(6, weight=10)

        #style = ttk.Style()
        #style.configure("Green.TLabel", foreground="white", background="green")

        self.createWidgets()
        """when initated, try open a json to get the default and output dirs. 
        if not found, use hardcoded defaults/os.getcwd().
        Make button so can change default dir and output dir and save the json. 
        Also button to reset to hardcoded defaults
        Same for matplotlib settings
        Make input (radio?) for potentiostat types: Autolab, Zahner, utgard smallones, other(contact me)
        make radio button to select files or folders
        Make exe icon / tk.root icon"""

    def createWidgets(self) -> None:
        self.some_mathButton = ttk.Button(self.master,
                                          text='Random Math',
                                          command=lambda: print(somecython.somemath(int(input("input number, press enter ")))))
        self.some_mathButton.grid(row=2, column=3)

        self.data_dirButton = ttk.Button(self.master,
                                         text='Select Data',
                                         command=self.get_datapath)
        self.data_dirButton.grid(row=3, column=3)

        self.goButton = ttk.Button(self.master,
                                   text='Go',
                                   command=self.mung_loop,
        )
        self.goButton.grid(row=4, column=3)

        self.quitButton = ttk.Button(self.master,
                                     text='Quit',
                                     command=self.quit, #style="Green.TLabel")
        )
        self.quitButton.grid(row=6, column=3, columnspan=1, ipady=10, ipadx=10)


    def mung_loop(self):
        #executor = ThreadPoolExecutor(max_workers=10)
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(open_mung_save(self.data_dir))
        except Exception as e:
            print(e)
        #fut = loop.run_in_executor(executor, open_mung_save, self.data_dir)
        # loop.close()
        # reveal_type(fut)
        # return fut

    def get_datapath(self) -> None:
        # new_data_dir: pathType = askdirectory(title='Select data folder')
        filetypes = [('All files', '*.*'), ('CSV files',
                                            '*.csv'), ('Text files', 'txt.*'), ]
        new_data_list: List[pathType] = askopenfilenames(
            title='Select data folder', filetypes=filetypes)
        if new_data_list is not None:
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
