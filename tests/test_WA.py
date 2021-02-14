from WA import water_added_to_pp


def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 4


def test_water_added_to_pp():
    x = water_added_to_pp([1, 2, 3])
    assert len(x) > 2
