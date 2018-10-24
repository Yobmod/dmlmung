#cython: language_level=3
import numpy as np
cimport numpy as cnp
import csv
cimport cython
from typing import Tuple, NamedTuple
from typing import Optional as Opt
from dml_thread.my_types import pathType, simpDict, paramsTup

from dml_thread import plot

@cython.ccall      
@cython.returns(cnp.ndarray[np.float64_t])
def open_file_numpy(data_dir: str, filename: str):
    """ """
    file = data_dir + "/" + filename
    if filename.endswith((".csv", )):   # ".txt")):
        with open(file, 'r') as source:
            data_array: np.ndarray = np.loadtxt(source,
                                            delimiter=",",
                                            skiprows=20,
                                            dtype=float)
        return data_array
    else:
        return None


@cython.ccall
@cython.returns(tuple)
def get_data_numpy(data_array: np.ndarray[np.float64_t]):
    x_array: Opt[np.ndarray] = None
    y_array: Opt[np.ndarray] = None
    imped: Opt[np.ndarray]
    phase: Opt[np.ndarray]
    if len(data_array[0]) == 4:
        x_array = data_array[:, [2]]  # pd
        y_array = data_array[:, [3]]  # current
    elif len(data_array[0]) == 6:
        freq = data_array[:, [1]]
        imped = data_array[:, [2]]
        phase = data_array[:, [3]]  # ; print(len(phase))
        # number = data_array[:, [0]]
        # signdif = data_array[:, [4]]
        # time = data_array[:, [5]]
        phase_rads = np.multiply(phase, (np.pi/180))
        freq_log = np.log10(freq)
        imped_log = np.log10(imped)
        phase_cos = np.cos(phase_rads)
        phase_sin = np.sin(phase_rads)
        imag_imped = np.absolute(np.multiply(imped, phase_sin, dtype=float))
        real_imped = np.multiply(imped, phase_cos, dtype=float)
    else:
        x_array = None  # np.empty(0) and remove Opt ? and is not Nones below
        y_array = None

    data: Opt[Tuple[np.ndarray, ...]]
    if x_array is not None and y_array is not None:  # both not falsey
        data = (x_array, y_array) # if x_array is not None else None
    elif imped is not None and phase is not None:
        data = (freq_log, imped_log, phase, imag_imped, real_imped)
    else:
        data = None
    return data

@cython.cfunc
def get_params(filename: str):  # Tuple[str, str, str, str, str]:
    """"""
    filename_strip = filename[:-4]
    if "agcl" in filename_strip:
        ref_elec = "Ag/AgCl"
    else:
        ref_elec = "Pt pseudo"

    if "50eu" in filename_strip:
        work_elec = "PuO2 + 50% Eu"
    elif "10eu" in filename_strip:
        work_elec = "PuO2 + 10% Eu"
    elif "puo2" in filename_strip:
        work_elec = "PuO2"
    else:
        work_elec = "Pt"

    if "naclo" in filename_strip:
        elec = "NaClO4"
    elif "tba" in filename_strip:
        elec = "TBA TFB"
    else:
        elec = "?"

    if "gbl" in filename_strip:
        solv = "GBL"
    elif "mecn" in filename_strip:
        solv = "MeCN"
    else:
        solv = "H2O"

    params: tuple = (filename_strip, solv, elec, ref_elec, work_elec)
    # reveal_type(params)
    return params


@cython.ccall
def write_imp_data(data: Tuple[np.ndarray, ...], params: paramsTup, output_dir: pathType=None) -> cython.void:
    """"""
    #(filename_strip, solv, elec, ref_elec, work_elec) = params
    filename_strip = params.filename_strip
    if len(data) == 5:
        (freq_log, imped_log, phase, imag_imped, real_imped) = data
        with open(f"{output_dir}/{filename_strip}.csv", 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            for x in range(len(freq_log)):
                csvwriter.writerow([float(freq_log[x]), 
                                    float(imped_log[x]), 
                                    float(phase[x]), 
                                    float(imag_imped[x]), 
                                    float(real_imped[x])])
            print(f"{filename_strip} impedance csv done")


@cython.ccall
def write_zview_data(data: Tuple[np.ndarray, ...], params: paramsTup, output_dir: pathType=None) -> cython.void:
    """""" 
    # (filename_strip, solv, elec, ref_elec, work_elec) = params
    filename_strip = params.filename_strip
    if len(data) == 5:
        (freq_log, imped_log, phase, imag_imped, real_imped) = data     # pylint: disable=W0612
        freq = 10 ** (freq_log)
        with open(f"{output_dir}/{filename_strip}_zveiw.csv", 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Z60W Data File: Version 1.1'])
            csvwriter.writerow([])
            csvwriter.writerow([])
            csvwriter.writerow([])
            csvwriter.writerow([])
            csvwriter.writerow([])
            csvwriter.writerow([])
            csvwriter.writerow([])
            csvwriter.writerow([0, 2, 0, 1, 0.1, 100000])
            csvwriter.writerow([78])
            for x in range(len(freq_log)):
                csvwriter.writerow([float(freq[x]), 0, 0, 0, float(real_imped[x]), float(-imag_imped[x]), 0, 0, 0])
            print(f"{filename_strip} zveiw csv done")



def fourier_smooth(x_var: np.ndarray, y_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """"""
    N = len(y_var)  # get the fourier transform
    y_hat = np.fft.rfft(y_var, n=N, axis=0)
    y_hat[10:] = 0.0
    y_smooth: np.ndarray = np.fft.irfft(y_hat, N, axis=0)
    return (x_var, y_smooth)


fontlab_default: simpDict = dict(
    family='sans-serif', color='darkred', weight='normal', size=12)


@cython.cfunc
def open_mung_save(data_dir: pathType,
                   filename: pathType,
                   output_dir: Opt[pathType] = None,
                   settings: Opt[simpDict] = None
                   ) -> cython.void:
    """"""
    cv_file = open_file_numpy(data_dir, filename)
    if cv_file is not None:
        data = get_data_numpy(cv_file)  # if cv_file is not None else None
        params = get_params(filename)
        if data is not None and len(data) == 2:
            (x_var, y_var) = data
            print(settings)
            plot.make_cv_plot(x_var, y_var, params,
                              output_dir, settings)
        elif data is not None and len(data) == 5:
            write_imp_data(data, params, output_dir)
            write_zview_data(data, params, output_dir)
            (freq_log, imped_log, phase, imag_imped, real_imped) = data
            plot.make_bode_plot(freq_log, imped_log, phase, params, output_dir)
            plot.make_nyquist_plot(imag_imped, real_imped, params, output_dir)


""""
def get_imp_numpy(data_array: np.ndarray) -> Opt[Tuple[np.ndarray, np.ndarray]]:
    if len(data_array[0]) == 6:
        freq = data_array[:, [1]]
        imped = data_array[:, [2]]
        phase = data_array[:, [3]]
        number = data_array[:, [0]]
        signif = data_array[:, [4]]
        time = data_array[:, [6]]
        phase_cos = np.cos(phase)
        phase_sin = np.sin(phase)
        imag_imped = np.multiply(data_array, phase_cos, dtype=float)
        real_imped = np.multiply(data_array, phase_cos, dtype=float)
    else:
        x_array = None
        y_array = None

    if x_array is not None and y_array is not None:  # both not falsey
        data: Opt[Tuple[np.ndarray, np.ndarray]] = (x_array, y_array)
    else:
        data = None
    return data
"""


if __name__ == "__main__":
    import os
    dd = os.getcwd()
    ee = pathType(f"{dd}/test")
    
    for ff in os.listdir(ee): 
        array = open_file_numpy(ee, ff)
        get_data_numpy(array)

"""
def open_file_get_data_numpymanual(data_dir: str, filename: str) -> Opt[Tuple[np.ndarray, np.ndarray]]:
    file = data_dir + "/" + filename
    line_num = 0
    if filename.endswith((".csv", ".txt")):
        with open(file, 'r') as f:
            for line in f:  # to get number of lines to prebuilt a fixed lenth & typed numpy array
                line_num += 1
            x_array = np.ndarray('line_num of floats', dtype=float)
            y_array = np.ndarray('line_num of floats', dtype=float)
            # line_num = len(list(f))

            for curr_line_num, line in enumerate(f):
                line.split(',')  # .split('\t')
                if len(line) == 2:
                    x_array[curr_line_num] = line[0]
                    y_array[curr_line_num] = line[1]
                else:
                    pass
        return (x_array, y_array)
    else:
        return None



import pandas as pd

def open_file(data_dir: str, filename: str) -> Opt[pd.DataFrame]:
    file = data_dir + "/" + filename
    if filename.endswith((".csv", ".txt")):
        cv_file = pd.read_csv(file, header=19, sep=",")
    elif filename.endswith(".tsv"):
        cv_file = pd.read_csv(file, header=19, sep="\t")
    else:
        cv_file = None
    return cv_file


def get_data(cv_file: pd.DataFrame) -> Opt[Tuple[pd.Series, pd.Series]]:
    if len(cv_file.columns) == 4:
        x_var = cv_file.iloc[:, 2:3]
        y_var = cv_file.iloc[:, 3:4] * 1000000  # converts to uA
    elif len(cv_file.columns) == 2:
        x_var = cv_file.iloc[:, 0:1]
        y_var = cv_file.iloc[:, 1:2] * 1000000  # converts to uA
    else:
        x_var = None
        y_var = None

    if x_var is not None and y_var is not None:  # both not falsey
        data: Opt[Tuple[pd.Series, pd.Series]] = (x_var, y_var)
    else:
        data = None
    return data


def open_file_get_data_python(data_dir: str, filename: str) -> Opt[Tuple[List[float], List[float]]]:
    file = data_dir + "/" + filename
    x_list = []
    y_list = []
    if filename.endswith((".csv", ".txt")):
        with open(file, 'r') as f:
            line = f.readline().split(',')  # .split('\t')
            if len(line) == 2:
                x_list.append(float(line[0]))
                y_list.append(float(line[1]))
            else:  # elif len(line) == 4:e
                pass
        return (x_list, y_list)
    else:
        return None
"""


"""
from zipfile import ZipFile, ZIP_DEFLATED

working_dir = os.getcwd()
dest_path = f"{working_dir}/dist"
zip_source_path = f"{working_dir}/dist/main"
zip_dest_path = f'{working_dir}/dist/main.zip'

with ZipFile(zip_dest_path, 'w', ZIP_DEFLATED) as myzip:
    for rootdir, subdirs, files in os.walk(zip_source_path):
        myzip.write(rootdir)
        for filename in files:
            myzip.write(os.path.join(rootdir, filename))

import shutil
shutil.make_archive(zip_source_path, 'gztar', dest_path)  # 'zip' 'bztar'
"""
