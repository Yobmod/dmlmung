from my_types import pathType, paramsTup
from typing import Dict, Mapping, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# from typing import Tuple  # , override, get_type_hints


data_dir: pathType = os.getcwd()  # cast(pathType, os.getcwd())


def get_json_settings(json_file: pathType) -> Dict[str, Union[str, int]]:
    """"""
    try:
        with open('./' + json_file, 'r') as jf:
            settings: Dict = json.load(jf)
    except Exception as e:
        settings = dict(family='sans-serif', color='darkred', weight='normal', size=12)
        print(e)
    return settings


def save_json_settings(json_file: pathType, setting_dict: Mapping) -> None:
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


def make_cv_plot(x_var: np.ndarray, y_var: np.ndarray, params: paramsTup, output_dir: str = None,
                 settings: Optional[Mapping] = None) -> None:
    """"""

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax2 = ax1.twiny()
    # fig.subplots_adjust(bottom=0.2)
    ax1.plot(x_var, y_var)
    xmin, xmax = plt.xlim()     # pylint: disable=W0612

    (filename_strip, solv, elec, ref_elec, work_elec) = params
    expt_vars = f"WE = {work_elec} \nRE = {ref_elec} \nElectrolyte = {elec} \nSolvent = {solv}"
    plt.title(
        f"CV of {work_elec} in {elec} ({solv}) vs {ref_elec} reference", y=1.05)

    plt.xlabel(
        f'Applied potential vs {ref_elec} reference (V)', fontdict=settings)

    plt.ylabel('Current' + ' (uA)', fontdict=settings)
    plt.text(xmax * 1.1, 0, expt_vars, fontdict=settings, withdash=False)

    plt.savefig(f"{output_dir}/{filename_strip}" + '_auto.png',
                bbox_inches='tight', dpi=500, transparent=True)

    plt.ylim(-1, 1)  # set axis limits
    plt.savefig(f"{output_dir}/{filename_strip}" + '_1uA.png',
                bbox_inches='tight', dpi=500, transparent=True)

    print(f"{filename_strip} cv plots done")
    # plt.show()


def make_nyquist_plot(
        imped_imag: np.ndarray, imped_real: np.ndarray, params: paramsTup, output_dir: pathType = None,
        settings: Optional[Mapping] = None) -> None:
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


def make_bode_plot(freq_log: np.ndarray, imped_log: np.ndarray, phase: np.ndarray, params: paramsTup,
                   output_dir: pathType = None, settings: Optional[Mapping] = None) -> None:
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
    plt.title(
        f"Bode plot of {work_elec} in {elec} ({solv}) vs {ref_elec} reference", y=1.05)

    plt.xlabel(f'Log Frequency (Hz)', fontdict=settings)

    plt.ylabel('Log Impedance (Ohms)', fontdict=settings)
    plt.text(xmax * 1.1, 0, expt_vars, fontdict=settings, withdash=False)

    plt.savefig(f"{output_dir}/{filename_strip}" + 'bode_auto.png',
                bbox_inches='tight', dpi=500, transparent=True)

    plt.savefig(f"{output_dir}/{filename_strip}" + 'bode_fixed.png',
                bbox_inches='tight', dpi=500, transparent=True)

    print(f"{filename_strip} bode plot done")
    # plt.show()


def multi_bode_axes(settings: Mapping[str, Union[str, int]], y_axes: str = "both") -> matplotlib.figure.Figure:
    """"""

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    plt.xlabel('Log Frequency', labelpad=10, fontdict=settings)

    phase_y_title = 'Phase (Theta)'
    gain_y_title = 'Log Impedance'

    if y_axes in ("phase", "p"):
        plt.ylim(-90, 0)
        plt.ylabel(phase_y_title, labelpad=10,  fontdict=settings)
    elif y_axes in ("gain", "g"):
        plt.ylabel(gain_y_title, labelpad=10, fontdict=settings)
    elif y_axes in ("both", "b"):
        plt.ylabel(gain_y_title, labelpad=10, fontdict=settings)
        _ = ax1.twinx()
        plt.ylim(-90, 0)
        plt.ylabel(phase_y_title, labelpad=10, fontdict=settings)
    else:
        raise ValueError('y_axes must be "both", "phase", or "gain" (or inital letters of.)')

    fig.params = None
    fig.voltages = []
    fig.y_axes = y_axes
    return fig


def multi_bode_title(fig, params, settings):

    # plt.figure(fig)  # make sure fig is focus
    (filename_strip, solv, elec, ref_elec, work_elec, voltage) = params
    elec = "TBFTFB"
    fig.suptitle(f"Bode plot of {work_elec} in {elec} ({solv}) vs {ref_elec} reference at {voltage} V", y=0.98)
    # xmin, xmax = plt.xlim()     # pylint: disable=W0612
    # expt_vars = f"WE = {work_elec} \nRE = {ref_elec} \nElectrolyte = {elec} \nSolvent = {solv}"
    # fig.text(xmax * 1.1, 0, expt_vars, fontdict=settings, withdash=False)
    fig.params = params
    return fig


def multi_bode_add_data(fig, freq_log: np.ndarray, imped_log: np.ndarray, phase: np.ndarray, params: paramsTup):
    axs = fig.axes
    voltage_label = f"{params.voltage: .2f} V" if params.voltage is not None else ""
    ax1 = axs[0]
    fig.voltages.append(params.voltage)

    if fig.y_axes == "both":
        ax1, ax2 = axs
        ax1.plot(freq_log, imped_log, label=voltage_label)
        ax2.plot(freq_log, phase)

    elif fig.y_axes == "gain":
        ax1.plot(freq_log, imped_log, label=voltage_label)

    elif fig.y_axes == "phase":
        ax1.plot(freq_log, phase, label=voltage_label)

    return fig


def multi_bode_plot_save(fig,
                         params: paramsTup,
                         output_dir: pathType = R".\output",
                         color_map="",
                         settings: Optional[Mapping] = None,
                         ) -> None:

    output_path = Path.cwd() / output_dir
    plt.figure(fig)  # make sure fig is focus
    axs = fig.axes

    handles, labels = axs[0].get_legend_handles_labels()
    combined = sorted(zip(fig.voltages, labels, handles))

    if color_map:
        color_map = plt.cm.get_cmap(color_map, len(fig.voltages))  # coolwarm twightlight
        colors = [color_map(i) for i in range(len(fig.voltages))]

        lines = axs[0].lines
        sorted_lines = sorted(zip(fig.voltages, lines, colors))

        for _volts, line, col in sorted_lines:
            line.set_color(col)
    legend_anchor = (1.05, 1)

    if len(axs) > 1:
        handles2, labels2 = axs[1].get_legend_handles_labels()
        combined2 = sorted(zip(fig.voltages, labels2, handles2))
        lines2 = axs[1].lines
        if color_map:
            sorted_lines2 = sorted(zip(fig.voltages, lines2, colors))
            for _volts, line, col in sorted_lines2:
                line.set_color(col)
        legend_anchor = (1.15, 1)

    axs[0].legend([handle for _, _, handle in combined],
                  [label for _, label, _ in combined],
                  bbox_to_anchor=legend_anchor, loc='upper left')

    filename_strip = params.filename_strip

    plt.savefig(output_path / f'{filename_strip}_{fig.y_axes}_multi_bode_auto.png',
                bbox_inches='tight', dpi=500, transparent=True)

    plt.xlim(-1.5, 5)
    plt.savefig(output_path / f'{filename_strip}_multi_bode_fixed.png',
                bbox_inches='tight', dpi=500, transparent=True)

    print(f"{filename_strip} multi bode plot done")
    # plt.show()


def multi_bode_plot(
        freq_log: np.ndarray, imped_log: np.ndarray, phase: np.ndarray, params: paramsTup,
        output_dir: pathType = None, settings: Optional[Mapping] = None, check_params=True) -> None:
    pass


def multi_nyquist_axes(settings) -> matplotlib.figure.Figure:
    """"""
    fig = plt.figure()
    _ = fig.add_subplot(111)
    plt.xlabel(f'Real Impedance (Ohms)', labelpad=10, fontdict=settings)
    plt.ylabel('Imaginary Impedance (Ohms)', labelpad=10, fontdict=settings)

    fig.params = None
    fig.voltages = []
    return fig


def multi_nyquist_title(fig: matplotlib.figure.Figure, params: paramsTup, settings) -> matplotlib.figure.Figure:

    # plt.figure(fig)  # make sure fig is focus
    (filename_strip, solv, elec, ref_elec, work_elec, voltage) = params
    elec = "TBFTFB"
    fig.suptitle(f"Nyquist plots of {work_elec} in {elec} ({solv}) vs {ref_elec} reference", y=0.98)
    # xmin, xmax = plt.xlim()     # pylint: disable=W0612
    # expt_vars = f"WE = {work_elec} \nRE = {ref_elec} \nElectrolyte = {elec} \nSolvent = {solv}"
    # fig.text(xmax * 1.1, 0, expt_vars, fontdict=settings, withdash=False)
    return fig


def multi_nyquist_add_data(
        fig, imped_imag: np.ndarray, imped_real: np.ndarray, params: paramsTup) -> matplotlib.figure.Figure:
    ax1 = fig.axes[0]
    voltage_label = f"{params.voltage: .2f} V" if params.voltage is not None else None
    ax1.plot(imped_real, imped_imag, label=voltage_label)

    fig.voltages.append(params.voltage)
    return fig


def multi_nyquist_plot_save(fig: matplotlib.figure.Figure,
                            params: paramsTup,
                            output_dir: pathType = R".\output",
                            set_colors="",
                            settings: Optional[Mapping] = None) -> None:

    output_path = Path.cwd() / output_dir
    plt.figure(fig)  # make sure fig is focus
    axs = fig.axes

    handles, labels = axs[0].get_legend_handles_labels()
    combined = sorted(zip(fig.voltages, labels, handles))
    if set_colors:
        color_map = plt.cm.get_cmap(set_colors, len(fig.voltages)+2)  # coolwarm twilight
        colors = [color_map(i) for i in range(len(fig.voltages)+2)][1::]

        lines = axs[0].lines
        sorted_lines = sorted(zip(fig.voltages, lines, colors))

        for _volts, line, col in sorted_lines:
            line.set_color(col)

    legend_anchor = (1.05, 1)
    axs[0].legend([handle for _, _, handle in combined],
                  [label for _, label, _ in combined],
                  bbox_to_anchor=legend_anchor,
                  loc='upper left',
                  )

    filename_strip = params.filename_strip

    plt.savefig(output_path / f'{filename_strip}_multi_nyquist_auto_{set_colors}.png',
                bbox_inches='tight', dpi=500, transparent=True)

    """plt.xlim(0, 25_000_000)
    plt.ylim(0, 60_000_000)
    plt.savefig(output_path / f'{filename_strip}_multi_nyquist_fixed.png',
                bbox_inches='tight', dpi=500, transparent=True)"""

    print(f"{filename_strip} multi nyquist plot done")


if __name__ == '__main__':
    pass
