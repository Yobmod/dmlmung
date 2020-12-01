from pathlib import Path
from WA import (table_bet_multi, water_added_to_pp,
                freq_change_to_WA,
                PlotInputTDict,
                WA_dataset,
                plot_multi,
                plot_bet_multi,
                )


x_ceo2_ox = water_added_to_pp(
    [10, 10, 10, 10,
    25, 25, 25, 25,
    50, 50, 50, 50,
    75, 75, 75, 75, 
    100, 100, 100, 100, 
    150, 150, 150, 150,
    200, 200, 200, 200, 
    240, 240,240, 240
    ])
y_ceo2_ox = freq_change_to_WA(
    [14.6, 11.1,  16.6, 9.1,
    19.8, 16.3,  21.8, 19.0,
    23.4, 21.5, 24.4, 22.5,
    28.8, 27.9, 30.8, 28.9,
    29.5, 36.6, 31.5, 30.6,
    43.2, 39.5, 32.2, 30.5,
    57.4, 61.3, 41.4, 44.3,
    90.2, 154.8, 52.2, 56.8],
    mass_mat=69, F0=5791070
    )


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

x_tho2_ox = water_added_to_pp(
    [10, 10, 25, 25, 50, 50, 75, 75, 100, 100, 150, 150, 200, 200, 240, 240])
y_tho2_ox = freq_change_to_WA(
    [17.0, 11.2, 
    20.15, 23.1, 
    26.5, 32.5, 
    29.5, 30.2, 34.2, 33.4, 38.3, 41.1, 63.1, 52.5, 82.9, 156.0],
    mass_mat=70.5, F0=5794505
)

oxal_data_500: PlotInputTDict = {
    "title": "MOx from oxalate @ 500°C",
    "data": [
        WA_dataset(x_ceo2_ox, y_ceo2_ox, "CeO\u2082 (oxalate)", color="c",mass=69, temp=75),
        WA_dataset(x_tho2_ox, y_tho2_ox, "ThO\u2082 (oxalate)", color="g",mass=70.5, temp=75),
        WA_dataset(x_uo2_ox, y_uo2_ox, "UO\u2082 (oxalate)", color="r",mass=55, temp=75),
    ],
}

sp = Path(R".\results")
oxalate_plot_500 = plot_multi(input=oxal_data_500, save_path=sp, file_name="plot_oxalate.png", fix_n=False, vmax_line=False)
oxalate_bet_500 = plot_bet_multi(input=oxal_data_500, save_path=sp, file_name="bet_oxalate.png", y_max=1e13)
oxalate_plot_500 = plot_multi(input=oxal_data_500, save_path=sp, file_name="plot_oxalate_fix.png", fix_n=True, vmax_line=False)
table_bet_multi(oxal_data_500, save_path=sp, file_name="bet_oxalate.tsv")