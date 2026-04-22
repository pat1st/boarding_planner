"""
optimizer.py — Steffen-based boarding sequence with group support.

The Steffen method boards passengers in this priority:
  1. Window seats (A, F)
  2. Middle seats (B, E)
  3. Aisle seats  (C, D)

Within each phase, alternating rows board first: the rows with an even
offset from the back of the plane board before those with an odd offset.
This lets multiple passengers stow baggage simultaneously without aisle
interference.

Groups (Option A — simple): a group is treated as a single unit and slotted
into the phase determined by its worst-positioned member (the member who
would board latest individually). The whole group boards together in that slot.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from models import Flight, Passenger, SeatType

# Phase order: earlier index = boards earlier
SEAT_TYPE_ORDER: List[SeatType] = [SeatType.WINDOW, SeatType.MIDDLE, SeatType.AISLE]


def _phase(seat_type: SeatType) -> int:
    return SEAT_TYPE_ORDER.index(seat_type)


def _steffen_key(passenger: Passenger, max_row: int) -> Tuple[int, int, int]:
    """
    Sort key for the Steffen sequence.  Lower key → boards earlier.

    Tuple: (phase, parity_from_back, -row)
      - phase          0=window, 1=middle, 2=aisle
      - parity         0 = even offset from back (boards first)
                       1 = odd  offset from back (boards second)
      - -row           larger row number (further back) boards first within
                       the same phase+parity group
    """
    phase = _phase(passenger.seat.seat_type)
    offset_from_back = max_row - passenger.seat.row
    parity = offset_from_back % 2
    return (phase, parity, -passenger.seat.row)


def _group_key(members: List[Passenger], max_row: int) -> Tuple[int, int, int]:
    """
    Sort key for a group: use the latest-boarding member's key so the whole
    group slots into the phase they would occupy last if boarding individually.
    """
    return max(_steffen_key(p, max_row) for p in members)


def compute_boarding_sequence(flight: Flight) -> List[List[Passenger]]:
    """
    Return an ordered list of boarding slots.

    Each slot is a list of passengers who board at the same time:
      - solo passengers: one per slot
      - groups: all members in a single slot

    The list is sorted by ascending sort key (earliest-boarding first).
    """
    max_row = flight.plane_config.rows

    groups: Dict[str, List[Passenger]] = defaultdict(list)
    solos: List[Passenger] = []

    for p in flight.passengers:
        if p.group_id:
            groups[p.group_id].append(p)
        else:
            solos.append(p)

    slots_with_keys: List[Tuple[Tuple, List[Passenger]]] = []

    for p in solos:
        slots_with_keys.append((_steffen_key(p, max_row), [p]))

    for group_id, members in groups.items():
        slots_with_keys.append((_group_key(members, max_row), list(members)))

    slots_with_keys.sort(key=lambda x: x[0])
    return [slot for _, slot in slots_with_keys]
