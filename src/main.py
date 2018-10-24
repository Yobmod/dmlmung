# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt

# from typing import List, Union, Dict, cast, NewType, Any  #, override, get_type_hints
# from typing import Optional as Opt
<<<<<<< HEAD
# from dmlechemmods.types import simpTypes, simpList, simpDict, compList, compDict, pathType # Num
DEBUG = True
=======
# from dmlechemmods.mytypes import simpTypes, simpList, simpDict, compList, compDict, pathType # Num
>>>>>>> d0972710d0b5140dd9c41fbd756ace6591dc3478


if __name__ == "__main__":
    # import pyximport
    # pyximport.install(pyimport=True)
    if DEBUG == True:
        import cProfile
        pr = cProfile.Profile() 
        pr.enable()

    import time
    start_time = time.perf_counter()
    import multiprocessing
    multiprocessing.freeze_support()
    print("starting...")
    # import PySide2
    # import os
    # import sys
    import tkinter as tk
    from dml_thread import gui  # , log  # , log, types, mung, plot, somecython
    from dml_thread.log import logger

    print(f"Imports done @ {time.perf_counter() - start_time:.2f} s")

    root = tk.Tk()
    root.iconbitmap(R'.\dml_thread\img\coffeebean.ico')
    root.geometry("700x400")  # wxh+x+y: as str
    root.title('DML E-Chem')
    app = gui.Application(master=root)

    try:
        print(f"GUI started @ {time.perf_counter() - start_time:.2f} s")
        app.mainloop()
    except (KeyboardInterrupt, SystemExit) as ke:
        print('\nApp closed with ctrl-C')
        logger.info(repr(ke))
    except Exception as e:
        logger.exception(repr(e))
        raise
    finally:
        print('DMLmung app closed')
        if DEBUG==True:
            pr.disable()
            import datetime
            import pstats
            now = datetime.datetime.now()
            stats = pstats.Stats(pr)
            stats.strip_dirs()
            stats.sort_stats('cumulative')
            stats.print_stats(15)
            filename = f'./logs/profile_{now.day}-{now.month}-{now.year}.prof'
            pr.dump_stats(filename)
            # pr.print_stats(sort='cumtime')
