from pathlib import Path
from WA import (table_bet_multi, water_added_to_pp,
                freq_change_to_WA,
                PlotInputTDict,
                WA_dataset,
                plot_multi,
                plot_bet_multi,
                )


x_ceo2_n = water_added_to_pp(
    [10, 10, 10,
    25, 25, 
    50, 50, 50,
    75, 75, 
    100, 100, 100,
    150, 150, 150,
    200, 200, 200,
    240, 240])
y_ceo2_n = freq_change_to_WA(
    [0.50* x for x in [
        10.4, 7.8, 9.9,
        19.0, 22.2, 
        23.5, 25.9, 22.4,
        24.6, 28.0, 
        33.4, 36.1, 24.6,
        35.8, 36.0, 25.6,
        47.4, 55.7, 32.8,
        80.0, 131.2]],
    mass_mat=46, F0=5751200
)


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

"""
x_tho2_n = water_added_to_pp(
    [10, 10, 25, 25, 50, 50, 75, 75, 100, 100, 150, 150, 200, 200, 240, 240])
y_tho2_n = freq_change_to_WA(
    [17.0, 11.2, 
    20.15, 23.1, 
    26.5, 32.5, 
    29.5, 30.2, 34.2, 33.4, 38.3, 41.1, 63.1, 52.5, 82.9, 156.0],
    mass_mat=70.5, F0=5794505
)
"""

nitrate_data_500: PlotInputTDict = {
    "title": "MOx from nitrate @ 500°C",
    "data": [
        WA_dataset(x_ceo2_n, y_ceo2_n, "CeO\u2082", color="c",mass=46, temp=75),
        # WA_dataset(x_tho2_n, y_tho2_n, "ThO\u2082", color="g",mass=70.5, temp=75),
        WA_dataset(x_uo2_n, y_uo2_n, "UO\u2082", color="r",mass=65, temp=75),
    ],
}

sp = Path(R".\results")
nitrate_plot_500 = plot_multi(input=nitrate_data_500, save_path=sp, file_name="plot_nitr.png", fix_n=False, vmax_line=False, equation=True)
oxalate_bet_500 = plot_bet_multi(input=nitrate_data_500, save_path=sp, file_name="bet_nitr.png", y_max=1e13)
oxalate_plot_500 = plot_multi(input=nitrate_data_500, save_path=sp, file_name="plot_nitr_fix.png", fix_n=True, vmax_line=False, , equation=False)
table_bet_multi(nitrate_data_500, save_path=sp, file_name="bet_nitr.tsv")