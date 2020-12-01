from pathlib import Path
from WA import plot_bet_multi, table_bet_multi, water_added_to_pp, freq_change_to_WA, PlotInputTDict, WA_dataset, plot_multi

x_uo2_ox = water_added_to_pp(
    [10, 10, 25, 25, 50, 50, 75, 75, 100, 100, 150, 150, 200, 200, 240, 240])
y_uo2_ox = freq_change_to_WA(
    [8.1, 6.4, 
    11.9, 12.3, 
    16.6, 14.2, 
    20.8, 17.9, 
    28.1, 22.7, 
    34.2, 37.5, 
    45.4, 39.3, 
    74.2, 111.5],
    mass_mat=55, F0=5798366)


x_uo2_n = water_added_to_pp(
    [10, 20, 25, 25, 50, 50, 100, 150, 240])
y_uo2_n = freq_change_to_WA(
    [4.5,  
    6.8, 
    12.5, 10.1, 
    16.3, 17.3, 
    20.8,
    25.1,
    100, 
    ],
    mass_mat=65, F0=5763900
)

urania_data_500: PlotInputTDict = {
    "title": "Urania from oxalate vs nitrate @ 500°C",
    "data": [
        WA_dataset(x_uo2_ox, y_uo2_ox, "UO\u2082 (oxalate)", color="r", mass=55, temp=75),
        WA_dataset(x_uo2_n, y_uo2_n, "UO\u2082 (nitrate)", color="m", mass=65, temp=75),
    ],
}

sp = Path(R".\results")
urania_plot_500 = plot_multi(input=urania_data_500,
                                save_path=sp,
                                file_name="plot_urania.png",
                                fix_n=False, vmax_line=False, equation=False)
urania_plot_500_fix = plot_multi(input=urania_data_500,
                                save_path=sp,
                                file_name="plot_urania_fix.png",
                                fix_n=True, vmax_line=False, equation=False)
urania_bet_500 = plot_bet_multi(input=urania_data_500, save_path=sp,
                                file_name="bet_urania.png", y_max=1e13)
table_bet_multi(urania_data_500, save_path=sp,file_name="bet_urania.tsv")