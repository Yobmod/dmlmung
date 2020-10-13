from __future__ import annotations
from typing import Dict, Optional, Union  # noqa: F401
import datetime
from pathlib import Path
from os import environ
import dataclasses
import myfitnesspal
from pprint import pprint
from dotenv import load_dotenv
import json

from typing_extensions import TypedDict


cwd = Path('.')
load_dotenv(dotenv_path=cwd / 'fp.env', verbose=True, encoding="UTF-8")

username = environ.get('fp_username')
password = environ.get('fp_password')
print(username, password)


class DailyTD(TypedDict):
    calories: int
    protein: int
    weight: Optional[float]


@ dataclasses.dataclass
class DailyData():
    calories: int
    protein: int
    weight: Optional[float] = None

    def as_dict(self) -> DailyTD:
        d = dataclasses.asdict(self)
        return DailyTD(calories=d['calories'], protein=d['protein'], weight=d['weight'])


client = myfitnesspal.Client(username, password)

aug2020 = datetime.date(2020, 8, 1)
today = datetime.date.today()
dates = today - aug2020

weights = client.get_measurements('Weight', aug2020, today)
start_weight = list(weights.values())[0]
current_weight = list(weights.values())[-1]

data: Dict[datetime.date, DailyData] = {}
total_protein = 0
total_cals = 0
food_logs = 0


for day in range(dates.days + 1):
    date = aug2020 + datetime.timedelta(days=day)

    fp_day = client.get_date(date)
    meals = fp_day.totals
    if meals:
        dd = DailyData(meals['calories'], meals['protein'])
        total_protein += meals['protein']
        total_cals += meals['calories']
        food_logs += 1
    else:
        dd = DailyData(0, 0)

    for x in weights:
        if x == date:
            dd.weight = weights[x]
    data.update({date: dd})

average_protein = total_protein / food_logs
average_cals = total_cals / food_logs
weight_lost = current_weight - start_weight

serialised_data: Dict[str, DailyTD] = {k.isoformat(): v.as_dict() for k, v in data.items()}

with open("fp.json", mode="w") as file:
    json.dump(serialised_data, file, indent=8)

pprint(data)
print(f"{average_protein}")
print(average_cals)
print(f"{weight_lost: .2f} kg")
