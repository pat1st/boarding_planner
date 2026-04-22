"""
scheduler.py — Assign estimated board_at times to each boarding slot.

Time estimates are based on baggage stow time per passenger.  Groups
partially parallelise (adjacent seats), so their slot duration is
max(member_times) + a small overhead per additional member.
"""
from __future__ import annotations

import datetime
from typing import List

import pandas as pd

from models import BaggageSize, Flight, Passenger
from optimizer import compute_boarding_sequence

# Estimated seconds for a passenger to stow baggage and sit down
BAGGAGE_SECONDS: dict = {
    BaggageSize.NONE: 8,
    BaggageSize.SMALL: 18,
    BaggageSize.LARGE: 35,
}

# Minimum gap between slots (walk-up / aisle-clearing time)
_GAP_SECONDS = 5


def slot_duration(passengers: List[Passenger]) -> int:
    """
    Estimate how long (seconds) a boarding slot will occupy the aisle.

    Solo:  baggage_time + gap
    Group: heaviest member's time + 5 s per additional member + gap
           (members stow partially in parallel at adjacent overhead bins)
    """
    times = [BAGGAGE_SECONDS[p.baggage_size] for p in passengers]
    if len(passengers) == 1:
        return times[0] + _GAP_SECONDS
    return max(times) + 5 * (len(passengers) - 1) + _GAP_SECONDS


def build_schedule(
    flight: Flight,
    boarding_start: datetime.datetime,
) -> pd.DataFrame:
    """
    Assign scheduled board_at times to every passenger and return a
    DataFrame with the full schedule (suitable for display and CSV export).
    """
    sequence = compute_boarding_sequence(flight)
    records = []
    current_time = boarding_start

    for slot_idx, slot in enumerate(sequence):
        duration = slot_duration(slot)
        for p in slot:
            p.scheduled_slot = slot_idx
            p.board_at = current_time
            records.append(
                {
                    "slot": slot_idx,
                    "board_at": current_time.strftime("%H:%M:%S"),
                    "passenger_id": p.passenger_id,
                    "name": p.name,
                    "seat": p.seat.seat_code,
                    "seat_type": p.seat.seat_type.value,
                    "group_id": p.group_id or "—",
                    "baggage": p.baggage_size.value,
                    "status": p.status.value,
                }
            )
        current_time += datetime.timedelta(seconds=duration)

    return pd.DataFrame(records)
