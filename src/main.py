"""."""
DEBUG = True

if DEBUG is True:
    import cProfile


def start_profiling() -> cProfile.Profile:
    import cProfile
    prof = cProfile.Profile()
    prof.enable()
    return prof


def prof_to_stats(profile: cProfile.Profile) -> None:
    """."""
    import pstats
    import datetime

    now = datetime.datetime.now()
    stats = pstats.Stats(profile)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(15)
    filename = f'./logs/profile_{now.day}-{now.month}-{now.year}.prof'
    profile.dump_stats(filename)


def main() -> None:
    """."""
    if DEBUG is True:
        prof = start_profiling()

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
    except (KeyboardInterrupt, SystemExit) as kexc:
        print('\nApp closed with ctrl-C')
        logger.info(repr(kexc))
    except Exception as exc:
        logger.exception(repr(exc))
        raise
    finally:
        print('DMLmung app closed')
        if DEBUG is True:
            prof.disable()
            prof_to_stats(prof)  # ; prof.print_stats(sort='cumtime')


if __name__ == "__main__":
    main()
