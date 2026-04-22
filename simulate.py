"""
simulate.py — Automated boarding simulation for evaluation.

Realistic late-passenger model:
  Passengers who miss their slot are removed from the queue entirely
  (matching the new gate.py behaviour).  Each no-show is then assigned
  a random delay (in slots) after which they "walk back to the gate"
  and handle_late_arrival() is called.  A configurable fraction of
  no-shows never return at all (permanent no-shows).

  Parameters
  ----------
  no_show_rate          fraction of passengers who miss their slot
  permanent_noshow_frac fraction of no-shows who never return (default 0.1)
  late_min_slots        minimum slots before a no-show returns (default 5)
  late_max_slots        maximum slots before a no-show returns (default 25)
"""
from __future__ import annotations

import copy
import datetime
import random
from dataclasses import dataclass
from typing import List, Optional

from gate import GateSession
from models import BoardingStatus, Flight, PlaneConfig
from scheduler import BAGGAGE_SECONDS, slot_duration


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SlotRecord:
    slot_index: int
    passenger_id: str
    name: str
    seat_code: str
    seat_type: str
    group_id: Optional[str]
    planned_slot: int           # slot in the original (no no-show) schedule
    actual_slot: int            # slot they actually boarded in
    planned_board_at: datetime.datetime
    actual_board_at: datetime.datetime
    was_noshow: bool


@dataclass
class SimResult:
    no_show_rate: float         # 0.0 – 1.0
    seed: int
    total_passengers: int
    no_show_count: int
    late_arrived_count: int     # no-shows who eventually returned
    permanent_noshow_count: int # no-shows who never returned
    baseline_seconds: int       # total time with 0 % no-shows
    total_seconds: int          # actual total boarding time
    time_overhead_pct: float    # (total - baseline) / baseline * 100
    avg_drift_seconds: float    # mean delay vs original schedule per pax
    phase_violations: int       # passengers who boarded in a later phase
    smoothness_score: float     # 0 (terrible) – 100 (perfect)
    records: List[SlotRecord]


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

# Maps passenger_id → the simulated clock time they actually boarded.
_ActualTimes = dict  # str → datetime.datetime

def _deep_copy_flight(flight: Flight) -> Flight:
    """Return a deep copy so repeated runs don't share mutable state."""
    return copy.deepcopy(flight)


def _phase_index(seat_type: str) -> int:
    order = ["window", "middle", "aisle"]
    return order.index(seat_type) if seat_type in order else 0


def run_simulation(
    flight: Flight,
    no_show_rate: float,
    seed: int = 42,
    boarding_start: Optional[datetime.datetime] = None,
    permanent_noshow_frac: float = 0.1,
    late_min_slots: int = 5,
    late_max_slots: int = 25,
) -> SimResult:
    """
    Simulate boarding for *flight* with *no_show_rate* fraction of
    passengers randomly missing their slot.

    Parameters
    ----------
    flight                : source Flight (not mutated; deep-copied internally)
    no_show_rate          : fraction 0.0–1.0 who miss their slot
    seed                  : RNG seed for reproducibility
    boarding_start        : simulation clock start; defaults to today 14:00
    permanent_noshow_frac : fraction of no-shows who never return (0–1)
    late_min_slots        : minimum slot-delay before a no-show returns
    late_max_slots        : maximum slot-delay before a no-show returns
    """
    if boarding_start is None:
        boarding_start = datetime.datetime.combine(
            datetime.date.today(), datetime.time(14, 0)
        )

    rng = random.Random(seed)

    # --- Baseline: 0 % no-shows, to normalise the time overhead ----------
    baseline_flight = _deep_copy_flight(flight)
    baseline_session = GateSession(baseline_flight, boarding_start)
    baseline_seconds, _ = _run_all_scanned(baseline_session)

    # --- Actual run -------------------------------------------------------
    sim_flight = _deep_copy_flight(flight)
    session = GateSession(sim_flight, boarding_start)

    all_ids = [p.passenger_id for p in sim_flight.passengers]
    n_noshow = round(len(all_ids) * no_show_rate)
    noshow_ids: set[str] = set(rng.sample(all_ids, n_noshow))

    # Decide which no-shows are permanent and which return after a delay.
    # Permanent no-shows: never call handle_late_arrival.
    # Returning no-shows: assigned a random return slot.
    n_permanent = round(n_noshow * permanent_noshow_frac)
    permanent_ids: set[str] = set(rng.sample(sorted(noshow_ids), n_permanent))
    returning_ids = noshow_ids - permanent_ids

    # Map each returning no-show to the slot index at which they arrive back.
    # Arrival slot = their original slot + random delay in [late_min, late_max].
    original_slot_of: dict[str, int] = {
        p.passenger_id: (p.scheduled_slot or 0)
        for p in sim_flight.passengers
    }
    arrival_slot_of: dict[str, int] = {
        pid: original_slot_of[pid] + rng.randint(late_min_slots, late_max_slots)
        for pid in returning_ids
    }
    # Set of passenger_ids still waiting to be triggered this run.
    pending_arrivals: set[str] = set(returning_ids)

    # Build lookups for drift / phase violation calculation.
    original_board_at: dict[str, datetime.datetime] = {
        p.passenger_id: p.board_at
        for p in sim_flight.passengers
        if p.board_at is not None
    }
    original_seat_type: dict[str, str] = {
        p.passenger_id: p.seat.seat_type.value
        for p in sim_flight.passengers
    }

    actual_board_times: dict[str, datetime.datetime] = {}
    records: List[SlotRecord] = []
    clock = boarding_start
    # Passengers who have physically returned and been requeued via
    # handle_late_arrival.  They must be SCANNED (not re-no-showed) when
    # their new slot is reached.
    requeued_ids: set[str] = set()

    while not session.is_complete:
        current_idx = session.current_slot_index

        # Trigger any returning no-shows whose arrival slot has been reached.
        arrived_now = [
            pid for pid in list(pending_arrivals)
            if arrival_slot_of[pid] <= current_idx
        ]
        for pid in arrived_now:
            session.handle_late_arrival(pid)
            requeued_ids.add(pid)
            pending_arrivals.discard(pid)

        current = session.get_current_call()
        if not current:
            break

        slot_clock = clock
        duration = slot_duration(current)

        for p in list(current):
            # Mark as no-show only if they haven't yet returned to the gate.
            if (p.passenger_id in noshow_ids
                    and p.passenger_id not in requeued_ids
                    and p.status == BoardingStatus.WAITING):
                session.mark_no_show(p.passenger_id)
            else:
                session.scan_passenger(p.passenger_id)
                actual_board_times[p.passenger_id] = slot_clock

        # Force the session to advance past any slot that is now fully
        # cleared (all scanned or all removed via no-show).  This guards
        # against edge cases where _try_advance was not called with the
        # correct slot contents.
        session._try_advance()

        clock += datetime.timedelta(seconds=duration)

    total_seconds = int((clock - boarding_start).total_seconds())

    # --- Build records for all scanned passengers -------------------------
    for p in sim_flight.passengers:
        if p.status != BoardingStatus.SCANNED:
            continue

        was_ns = p.passenger_id in noshow_ids
        orig_slot = original_slot_of.get(p.passenger_id, p.scheduled_slot or 0)
        orig_bat  = original_board_at.get(p.passenger_id, boarding_start)
        actual_bat = actual_board_times.get(p.passenger_id, clock)

        records.append(SlotRecord(
            slot_index=p.scheduled_slot or 0,
            passenger_id=p.passenger_id,
            name=p.name,
            seat_code=p.seat.seat_code,
            seat_type=p.seat.seat_type.value,
            group_id=p.group_id,
            planned_slot=orig_slot,
            actual_slot=p.scheduled_slot or 0,
            planned_board_at=orig_bat,
            actual_board_at=actual_bat,
            was_noshow=was_ns,
        ))

    # --- Metrics ----------------------------------------------------------
    if records:
        drifts = [
            max(0.0, (r.actual_board_at - r.planned_board_at).total_seconds())
            for r in records
        ]
        avg_drift = sum(drifts) / len(drifts)
        phase_violations = sum(
            1 for r in records
            if _phase_index(r.seat_type) <
               _phase_index(original_seat_type.get(r.passenger_id, r.seat_type))
        )
    else:
        avg_drift = 0.0
        phase_violations = 0

    overhead_pct = (
        (total_seconds - baseline_seconds) / baseline_seconds * 100
        if baseline_seconds > 0 else 0.0
    )
    overhead_penalty = min(overhead_pct, 100.0)
    drift_penalty    = min(avg_drift / 60.0 * 10, 50.0)
    smoothness_score = max(0.0, 100.0 - overhead_penalty * 0.5 - drift_penalty)

    late_arrived = len(returning_ids) - len(pending_arrivals)  # arrived before boarding ended

    return SimResult(
        no_show_rate=no_show_rate,
        seed=seed,
        total_passengers=len(sim_flight.passengers),
        no_show_count=n_noshow,
        late_arrived_count=late_arrived,
        permanent_noshow_count=len(permanent_ids) + len(pending_arrivals),  # permanent + didn't make it back in time
        baseline_seconds=baseline_seconds,
        total_seconds=total_seconds,
        time_overhead_pct=round(overhead_pct, 1),
        avg_drift_seconds=round(avg_drift, 1),
        phase_violations=phase_violations,
        smoothness_score=round(smoothness_score, 1),
        records=records,
    )


def run_sweep(
    flight: Flight,
    rates: Optional[List[float]] = None,
    seed: int = 42,
    boarding_start: Optional[datetime.datetime] = None,
    permanent_noshow_frac: float = 0.1,
    late_min_slots: int = 5,
    late_max_slots: int = 25,
) -> List[SimResult]:
    """
    Run run_simulation for each rate in *rates* and return all results.
    Default sweep: 0 %, 5 %, 10 %, …, 80 %
    """
    if rates is None:
        rates = [r / 100 for r in range(0, 85, 5)]
    return [
        run_simulation(
            flight, rate, seed=seed + i,
            boarding_start=boarding_start,
            permanent_noshow_frac=permanent_noshow_frac,
            late_min_slots=late_min_slots,
            late_max_slots=late_max_slots,
        )
        for i, rate in enumerate(rates)
    ]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _run_all_scanned(session: GateSession) -> tuple[int, dict]:
    """Scan every passenger in order; return (total_elapsed_seconds, actual_board_times)."""
    start = session.boarding_start
    clock = start
    actual_times: dict[str, datetime.datetime] = {}
    while not session.is_complete:
        current = session.get_current_call()
        if not current:
            break
        slot_clock = clock
        duration = slot_duration(current)
        for p in list(current):
            session.scan_passenger(p.passenger_id)
            actual_times[p.passenger_id] = slot_clock
        clock += datetime.timedelta(seconds=duration)
    return int((clock - start).total_seconds()), actual_times
