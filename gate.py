"""
gate.py — Live gate session: scan events and on-arrival reoptimisation.

Late-passenger model (realistic):
  When a passenger misses their slot they are removed from the queue
  entirely and held in NO_SHOW state.  They are NOT automatically
  requeued — in reality nobody knows when (or whether) they will arrive.

  When they physically appear at the gate the agent clicks "Arrived",
  which calls handle_late_arrival().  Only then is the passenger
  inserted at the best remaining slot (same boarding phase if still
  active, otherwise the next open slot).

  Passengers who never appear remain as permanent no-shows and do not
  board.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple

from models import BoardingStatus, Flight, Passenger, SeatType
from optimizer import SEAT_TYPE_ORDER, compute_boarding_sequence
from scheduler import slot_duration


@dataclass
class GateEvent:
    timestamp: datetime.datetime
    event_type: str   # "scanned" | "no_show" | "late_arrival"
    passenger_id: str
    message: str


class GateSession:
    """
    Tracks real-time boarding state for a single flight.

    The internal sequence is a mutable list-of-lists.  Slots may be
    inserted or deleted as the session progresses; _rebuild_times()
    keeps board_at and scheduled_slot consistent after every change.
    """

    def __init__(self, flight: Flight, boarding_start: datetime.datetime) -> None:
        self.flight = flight
        self.boarding_start = boarding_start
        self._sequence: List[List[Passenger]] = compute_boarding_sequence(flight)
        self._current_slot: int = 0
        self._events: List[GateEvent] = []
        self._rebuild_times()

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def current_slot_index(self) -> int:
        return self._current_slot

    @property
    def total_slots(self) -> int:
        return len(self._sequence)

    @property
    def is_complete(self) -> bool:
        return self._current_slot >= len(self._sequence)

    @property
    def events(self) -> List[GateEvent]:
        return list(self._events)

    # ------------------------------------------------------------------
    # Gate actions
    # ------------------------------------------------------------------

    def get_current_call(self) -> Optional[List[Passenger]]:
        """Passengers in the active slot, or None if boarding is done."""
        if self.is_complete:
            return None
        return self._sequence[self._current_slot]

    def get_upcoming_slots(self, n: int = 3) -> List[List[Passenger]]:
        """Preview of the next n slots after the current one."""
        start = self._current_slot + 1
        return self._sequence[start : start + n]

    def scan_passenger(self, passenger_id: str) -> Tuple[bool, str]:
        """Mark a passenger as boarded and advance the slot if complete."""
        p = self._find(passenger_id)
        if p is None:
            return False, f"Passenger {passenger_id} not found."
        if p.status == BoardingStatus.SCANNED:
            return False, f"{p.name} is already boarded."

        p.status = BoardingStatus.SCANNED
        p.scanned_at = datetime.datetime.now()
        self._log("scanned", passenger_id, f"{p.name} boarded — seat {p.seat.seat_code}")
        self._try_advance()
        return True, f"{p.name} boarded successfully."

    def mark_no_show(self, passenger_id: str) -> Tuple[bool, str]:
        """
        Passenger missed their slot.

        Removes them from the queue entirely and sets status to NO_SHOW.
        They will only re-enter the queue if/when handle_late_arrival()
        is called — i.e. when they physically show up at the gate.
        """
        p = self._find(passenger_id)
        if p is None:
            return False, f"Passenger {passenger_id} not found."
        if p.status == BoardingStatus.SCANNED:
            return False, f"{p.name} is already boarded."

        p.status = BoardingStatus.NO_SHOW
        self._remove_from_sequence(p)
        self._try_advance()
        self._rebuild_times()
        self._log(
            "no_show",
            passenger_id,
            f"{p.name} missed slot — waiting for gate arrival",
        )
        return True, f"{p.name} removed from queue. Click Arrived when they show up."

    def handle_late_arrival(self, passenger_id: str) -> Tuple[bool, str]:
        """
        Passenger has physically arrived at the gate after missing their slot.

        Inserts them at the best remaining slot (same boarding phase if still
        active, otherwise the next open slot) and sets status back to WAITING.
        """
        p = self._find(passenger_id)
        if p is None:
            return False, f"Passenger {passenger_id} not found."
        if p.status == BoardingStatus.SCANNED:
            return False, f"{p.name} is already boarded."
        if p.status == BoardingStatus.WAITING:
            return False, f"{p.name} is already in the queue."

        self._requeue(p)
        p.status = BoardingStatus.WAITING
        self._rebuild_times()
        slot_time = p.board_at.strftime("%H:%M:%S") if p.board_at else "TBD"
        self._log(
            "late_arrival",
            passenger_id,
            f"{p.name} arrived — requeued to slot {p.scheduled_slot} ({slot_time})",
        )
        return True, f"{p.name} requeued to slot {p.scheduled_slot}, estimated {slot_time}."

    def stats(self) -> dict:
        pax = self.flight.passengers
        total = len(pax)
        boarded = sum(1 for p in pax if p.status == BoardingStatus.SCANNED)
        no_show = sum(1 for p in pax if p.status == BoardingStatus.NO_SHOW)
        elapsed = int((datetime.datetime.now() - self.boarding_start).total_seconds())
        return {
            "total": total,
            "boarded": boarded,
            "remaining": total - boarded,
            "no_show": no_show,
            "compliance_pct": round(boarded / total * 100, 1) if total else 0.0,
            "slots_done": self._current_slot,
            "slots_total": len(self._sequence),
            "elapsed_seconds": elapsed,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_times(self) -> None:
        """
        Recompute board_at and scheduled_slot for every slot/passenger
        from _current_slot onward.  Already-boarded passengers keep their
        actual scan time; only waiting/no-show passengers are updated.
        """
        current_time = self.boarding_start
        for slot_idx, slot in enumerate(self._sequence):
            for p in slot:
                p.scheduled_slot = slot_idx
                if slot_idx >= self._current_slot and p.status != BoardingStatus.SCANNED:
                    p.board_at = current_time
            current_time += datetime.timedelta(seconds=slot_duration(slot))

    def _try_advance(self) -> None:
        """Advance _current_slot as long as all passengers in it are scanned."""
        while self._current_slot < len(self._sequence):
            slot = self._sequence[self._current_slot]
            if all(p.status == BoardingStatus.SCANNED for p in slot):
                self._current_slot += 1
            else:
                break

    def _remove_from_sequence(self, passenger: Passenger) -> None:
        """Remove a passenger from their slot; drop the slot if it empties."""
        for slot in self._sequence:
            if passenger in slot:
                slot.remove(passenger)
                if not slot:
                    idx = self._sequence.index(slot)
                    self._sequence.remove(slot)
                    # If a slot before the current pointer was somehow removed,
                    # keep the pointer consistent (in practice this won't happen
                    # since no-shows are always in the current or future slot).
                    if idx < self._current_slot:
                        self._current_slot = max(0, self._current_slot - 1)
                break

    def _requeue(self, passenger: Passenger) -> None:
        """
        Insert the passenger at the best remaining slot.

        Called only from handle_late_arrival(), so the passenger has been
        absent from the queue for some time.  We scan from _current_slot
        onward and insert before the first slot whose average boarding
        phase is >= the passenger's own phase.  This keeps phase ordering
        intact where possible; if their phase has passed they board ASAP.
        """
        pax_phase = SEAT_TYPE_ORDER.index(passenger.seat.seat_type)
        insert_idx = len(self._sequence)  # default: append at end

        for i in range(self._current_slot, len(self._sequence)):
            slot = self._sequence[i]
            if not slot:
                continue
            avg_phase = sum(
                SEAT_TYPE_ORDER.index(p.seat.seat_type) for p in slot
            ) / len(slot)
            if avg_phase >= pax_phase:
                insert_idx = i
                break

        self._sequence.insert(insert_idx, [passenger])
        # _rebuild_times() called by the caller will fix scheduled_slot / board_at

    def _find(self, passenger_id: str) -> Optional[Passenger]:
        for p in self.flight.passengers:
            if p.passenger_id == passenger_id:
                return p
        return None

    def _log(self, event_type: str, passenger_id: str, message: str) -> None:
        self._events.append(
            GateEvent(
                timestamp=datetime.datetime.now(),
                event_type=event_type,
                passenger_id=passenger_id,
                message=message,
            )
        )
