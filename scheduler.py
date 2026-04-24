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

# Per-person aisle-entry overhead for a Steffen wave.
# Passengers in a wave are 2 rows apart so bin stowing is parallel;
# this small constant represents each person walking to their row.
_WAVE_ENTRY_SECONDS = 2

# Fixed epoch used whenever we build a schedule outside a live gate session.
# All displayed times are relative offsets from this point, so the absolute
# date/hour is irrelevant — only the elapsed deltas matter.
SCHEDULE_EPOCH = datetime.datetime(2000, 1, 1, 0, 0, 0)


def slot_duration(passengers: List[Passenger]) -> int:
    """
    Estimate how long (seconds) a boarding slot will occupy the aisle.

    Solo:        baggage_time + gap
    Steffen wave (all solo pax, no group_id): max(baggage_times)
                 + WAVE_ENTRY_SECONDS per additional person + gap.
                 Passengers are 2 rows apart so bin stowing is parallel;
                 the small per-person overhead is aisle-walk only.
    Family group (shared group_id): max(baggage_times)
                 + 5 s per additional member + gap.
                 Members share an overhead bin zone so stowing is
                 partially sequential.
    """
    times = [BAGGAGE_SECONDS[p.baggage_size] for p in passengers]
    if len(passengers) == 1:
        return times[0] + _GAP_SECONDS
    is_steffen_wave = all(p.group_id is None for p in passengers)
    overhead = _WAVE_ENTRY_SECONDS if is_steffen_wave else 5
    return max(times) + overhead * (len(passengers) - 1) + _GAP_SECONDS


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
            _elapsed = int((current_time - boarding_start).total_seconds())
            records.append(
                {
                    "slot": slot_idx,
                    "board_at": f"+{_elapsed // 60}m {_elapsed % 60:02d}s",
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
