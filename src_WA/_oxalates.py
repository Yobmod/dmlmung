from WA import (water_added_to_pp,
                freq_change_to_WA,
                PlotInputTDict,
                WA_dataset,
                plot_multi,
                plot_bet_multi,
                )

# ThO2
x_ceo2_ox = water_added_to_pp(
    [10, 10, 25, 25, 50, 50, 75, 75, 100, 100, 150, 150, 200, 200, 240, 240])
y_ceo2_ox = freq_change_to_WA(
    [146, 91, 198, 170, 224, 205, 288, 269, 295, 286, 312, 295, 374, 403, 502, 548],
    mass_mat=69, F0=5791070)

x_tho2_ox = water_added_to_pp(
    [10, 10, 25, 25, 50, 50, 75, 75, 100, 100, 150, 150, 200, 200, 240, 240])
y_tho2_ox = freq_change_to_WA(
    [180, 180, 211.5, 211, 275, 275, 305, 305, 311.5, 311, 333.5, 333, 431.5, 431, 529, 540],
    mass_mat=70.5, F0=5794505
)

oxal_data_500: PlotInputTDict = {
    "title": "Ceria vs Thoria (from oxalate)",
    "data": [
        WA_dataset(x_ceo2_ox, y_ceo2_ox, "CeO\u2082 (oxalate)", color="c"),
        WA_dataset(x_tho2_ox, y_tho2_ox, "ThO\u2082 (oxalate)", color="g"),
    ],
}

oxalate_plot_500 = plot_multi(input=oxal_data_500, file_name="plot_oxalate.png", fix_n=True, vmax_line=True)
oxalate_bet_500 = plot_bet_multi(input=oxal_data_500, file_name="bet_oxalate.png")
