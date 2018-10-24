import os
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from typing import Tuple #, override, get_type_hints
from typing import Optional as Opt
from dml_thread.my_types import simpDict, pathType, paramsTup


data_dir: pathType = os.getcwd()  # cast(pathType, os.getcwd())


def get_json_settings(json_file: pathType) -> simpDict:
    """"""
    with open('./' + json_file, 'r') as jf: 
        settings: simpDict = json.load(jf)
    return settings


def save_json_settings(json_file: pathType, setting_dict: simpDict) -> None:
    """  """
    try:
        setting_json = json.dumps(setting_dict)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(e)
    else:
        with open(json_file, 'w') as jf:
            jf.write(setting_json)

"""
fontlab_default: simpDict = dict(family='sans-serif', color='darkred', weight='normal', size=12)

fontlab: Opt[simpDict] = get_json_settings('settings\settings.json')
if fontlab is None:
    fontlab = fontlab_default
# print(fontlab)
"""

def make_cv_plot(x_var: np.ndarray, y_var: np.ndarray, params: paramsTup, output_dir: str = None,  settings: Opt[simpDict] = None) -> None:
    """"""

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax2 = ax1.twiny()
    # fig.subplots_adjust(bottom=0.2)
    ax1.plot(x_var, y_var)
    xmin, xmax = plt.xlim()     # pylint: disable=W0612

    (filename_strip, solv, elec, ref_elec, work_elec) = params
    expt_vars = f"WE = {work_elec} \nRE = {ref_elec} \nElectrolyte = {elec} \nSolvent = {solv}"
    plt.title(f"CV of {work_elec} in {elec} ({solv}) vs {ref_elec} reference", y=1.05)

    plt.xlabel(f'Applied potential vs {ref_elec} reference (V)', fontdict=settings)

    plt.ylabel('Current' + ' (uA)', fontdict=settings)
    plt.text(xmax * 1.1, 0, expt_vars, fontdict=settings, withdash=False)

    plt.savefig(f"{output_dir}/{filename_strip}" + '_auto.png', bbox_inches='tight', dpi=500, transparent=True)

    plt.ylim(-1, 1)  # set axis limits
    plt.savefig(f"{output_dir}/{filename_strip}" + '_1uA.png', bbox_inches='tight', dpi=500, transparent=True)

    print(f"{filename_strip} cv plots done")
    # plt.show()


def make_nyquist_plot(imped_imag: np.ndarray, imped_real: np.ndarray, params: paramsTup, output_dir: pathType = None, settings: Opt[simpDict] = None) -> None:
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # imped_imag = np.absolute(imped_imag)
    ax1.plot(imped_real, imped_imag, 'b')
    # xmin, xmax = plt.xlim()
    filename_strip = params[0]
    # (filename_strip, solv, elec, ref_elec, work_elec) = params
    # expt_vars = f"WE = {work_elec} \nRE = {ref_elec} \nElectrolyte = {elec} \nSolvent = {solv}"
    # plt.title(f"Nyquist plot of {work_elec} in {elec} ({solv}) vs {ref_elec} reference", y=1.05)
    plt.xlabel(f'Real Impedance (Ohms)', fontdict=settings)
    plt.ylabel('Imaginary Impedance (Ohms)', fontdict=settings)
    plt.savefig(f'{output_dir}/{filename_strip}' + 'nyquist_auto.png',
                bbox_inches='tight', dpi=500, transparent=True)

    plt.xlim(0, 25_000_000)
    plt.ylim(0, 60_000_000)
    plt.savefig(f'{output_dir}/{filename_strip}' + 'nyquist_fixed.png',
                bbox_inches='tight', dpi=500, transparent=True)
    print(f"{filename_strip} nyquist plots done")

# import numba
# @numba.jit
def make_bode_plot(freq_log: np.ndarray, imped_log: np.ndarray, phase: np.ndarray, params: paramsTup, output_dir: pathType = None, settings: Opt[simpDict] = None) -> None:
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    xmin, xmax = plt.xlim()     # pylint: disable=W0612
    plt.ylim(3, 8)
    # fig.subplots_adjust(bottom=0.2)
    ax1.plot(freq_log, imped_log, 'b')

    ax2 = ax1.twinx()
    plt.ylim(-90, 0)
    ax2.plot(freq_log, phase, 'r-')

    (filename_strip, solv, elec, ref_elec, work_elec) = params
    expt_vars = f"WE = {work_elec} \nRE = {ref_elec} \nElectrolyte = {elec} \nSolvent = {solv}"
    plt.title(f"Bode plot of {work_elec} in {elec} ({solv}) vs {ref_elec} reference", y=1.05)

    plt.xlabel(f'Log Frequency (Hz)', fontdict=settings)

    plt.ylabel('Log Impedance (Ohms)', fontdict=settings)
    plt.text(xmax * 1.1, 0, expt_vars, fontdict=settings, withdash=False)

    plt.savefig(f"{output_dir}/{filename_strip}" + 'bode_auto.png', bbox_inches='tight', dpi=500, transparent=True)


    plt.savefig(f"{output_dir}/{filename_strip}" + 'bode_fixed.png',
                bbox_inches='tight', dpi=500, transparent=True)

    print(f"{filename_strip} bode plot done")
    # plt.show()
