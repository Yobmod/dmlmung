from numba import jit
from numba.typed import List as NBList
import numpy as np
from typing import List


@jit(nopython=True)
def water_added_to_pp_refl(water_added: List[int], vessel_volume: int = 1000
                           ) -> np.ndarray:
    """convert list of amounts of water in mg to array of partial pressures for a given volume"""
    wadd_array = np.asarray(water_added)
    print(wadd_array)
    return np.divide(wadd_array, vessel_volume)


@jit(nopython=True)
def water_added_to_pp_NBList(water_added: List[int], vessel_volume: int = 1000
                             ) -> np.ndarray:
    """convert list of amounts of water in mg to array of partial pressures for a given volume"""
    water_list = NBList()
    [water_list.append(x) for x in water_added]
    wadd_array = np.asarray(water_list)
    print(wadd_array)
    return np.divide(wadd_array, vessel_volume)


# output_reflected = water_added_to_pp_refl([10, 25, 50, 75, 100, 150, 200, 250])
output_NBListed = water_added_to_pp_NBList([10, 25, 50, 75, 100, 150, 200, 250])
