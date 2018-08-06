import time
start_time = time.perf_counter()
print("starting...")

# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt

# from typing import List, Union, Dict, cast, NewType, Any  #, override, get_type_hints
# from typing import Optional as Opt
# from dmlechemmods.types import simpTypes, simpList, simpDict, compList, compDict, pathType # Num

import tkinter as tk
from dml_async import gui  # , log # , types, mung, plot, somecython

print(f"Imports done @ {round(time.perf_counter() - start_time, 2):.2f} s")

import asyncio
async def run_tk(root: tk.Tk, interval: float=0.05) -> None:
    try:
        while True:
            root.update()
            await asyncio.sleep(interval)
    except tk.TclError as e:
        if "application has been destroyed" not in e.args[0]:
            raise

if __name__ == "__main__":
    root = tk.Tk()
    app = gui.Application(master=root)
    root.title('DML E-Chem')
    root.geometry("400x400")  # hxw+x+y: str
    print(f"GUI started @ {time.perf_counter() - start_time:.2f} s")
    # asyncio.get_event_loop().run_until_complete(run_tk(root))
    app.mainloop()

