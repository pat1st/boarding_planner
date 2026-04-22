"""
data_gen.py — Reproducible flight manifest generation and CSV loading.
"""
from __future__ import annotations

import random
from typing import List, Optional

import pandas as pd

from models import BaggageSize, Passenger, PlaneConfig, Seat

_FIRST_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Nick", "Olivia", "Paul",
    "Quinn", "Rose", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zane", "Anna", "Ben", "Carla", "Dan",
]
_LAST_NAMES = [
    "Smith", "Jones", "Brown", "Wilson", "Taylor", "Davies", "Evans",
    "Thomas", "Roberts", "Johnson", "Walker", "Wright", "Thompson", "White",
    "Hall", "Harris", "Lewis", "Clark", "Young", "King",
]


def generate_flight_csv(
    plane_config: PlaneConfig,
    group_ratio: float = 0.30,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a reproducible flight manifest.

    Parameters
    ----------
    plane_config : PlaneConfig
    group_ratio  : fraction of seats assigned to travel groups (0–1)
    seed         : random seed for full reproducibility
    output_path  : if given, writes the CSV to this path

    Returns
    -------
    pd.DataFrame with columns:
        passenger_id, name, row, column, group_id, baggage_size
    """
    rng = random.Random(seed)

    # Shuffle all seats so groups aren't guaranteed to share a row
    all_seats = [
        (row, col)
        for row in range(1, plane_config.rows + 1)
        for col in plane_config.columns
    ]
    rng.shuffle(all_seats)

    total = len(all_seats)
    group_pool = int(total * group_ratio)

    # Assign groups in blocks of 2–4 consecutive shuffled seats
    group_assignments: dict[int, str] = {}
    seat_idx = 0
    group_counter = 1
    while seat_idx < group_pool:
        size = rng.randint(2, 4)
        gid = f"G{group_counter:03d}"
        group_counter += 1
        for _ in range(size):
            if seat_idx >= group_pool:
                break
            group_assignments[seat_idx] = gid
            seat_idx += 1

    records = []
    for i, (row, col) in enumerate(all_seats):
        name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
        baggage = rng.choices(
            [BaggageSize.NONE.value, BaggageSize.SMALL.value, BaggageSize.LARGE.value],
            weights=[10, 60, 30],
        )[0]
        records.append(
            {
                "passenger_id": f"P{i + 1:04d}",
                "name": name,
                "row": row,
                "column": col,
                "group_id": group_assignments.get(i, ""),
                "baggage_size": baggage,
            }
        )

    df = pd.DataFrame(records)
    if output_path:
        df.to_csv(output_path, index=False)
    return df


def load_passengers_from_df(df: pd.DataFrame, plane_config: Optional[PlaneConfig] = None) -> List[Passenger]:
    """
    Convert a manifest DataFrame into a list of Passenger objects.

    Expects columns: passenger_id, name, row, column, group_id, baggage_size

    plane_config is used to resolve the correct SeatType for each column.
    Defaults to a standard A320 layout if omitted.
    """
    if plane_config is None:
        plane_config = PlaneConfig.a320()
    passengers: List[Passenger] = []
    for _, row in df.iterrows():
        col = str(row["column"]).upper()
        seat = Seat(
            row=int(row["row"]),
            column=col,
            seat_type=plane_config.seat_type_for(col),
        )
        passengers.append(
            Passenger(
                passenger_id=str(row["passenger_id"]),
                name=str(row["name"]),
                seat=seat,
                group_id=str(row["group_id"]) if row["group_id"] else None,
                baggage_size=BaggageSize(str(row["baggage_size"]).lower()),
            )
        )
    return passengers
