# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt

# from typing import List, Union, Dict, cast, NewType, Any  #, override, get_type_hints
# from typing import Optional as Opt
# from dmlechemmods.types import simpTypes, simpList, simpDict, compList, compDict, pathType # Num



if __name__ == "__main__":
    # import pyximport
    # pyximport.install(pyimport=True)
    import time
    start_time = time.perf_counter()
    import multiprocessing
    multiprocessing.freeze_support()
    print("starting...")
    import tkinter as tk
    from dml_thread import gui  # , log, types, mung, plot, somecython
    print(f"Imports done @ {time.perf_counter() - start_time:.2f} s")

    root = tk.Tk()
    app = gui.Application(master=root)
    #root.title('DML E-Chem')
    root.iconbitmap(R'.\dml_thread\img\coffeebean.ico')
    root.geometry("600x400")  # wxh+x+y: str
    print(f"GUI started @ {time.perf_counter() - start_time:.2f} s")

    app.mainloop()
