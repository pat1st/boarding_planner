"""
models.py — Core data classes for the boarding planner.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class BaggageSize(str, Enum):
    NONE = "none"
    SMALL = "small"
    LARGE = "large"


class SeatType(str, Enum):
    WINDOW = "window"
    MIDDLE = "middle"
    AISLE = "aisle"


class BoardingStatus(str, Enum):
    WAITING = "waiting"    # Not yet called
    SCANNED = "scanned"    # Ticket scanned — boarded
    NO_SHOW = "no_show"    # Missed their slot (may still arrive)


@dataclass
class Seat:
    row: int
    column: str  # A–K (varies by aircraft)
    seat_type: SeatType = field(default=SeatType.AISLE)

    @property
    def seat_code(self) -> str:
        return f"{self.row}{self.column}"


@dataclass
class Passenger:
    passenger_id: str
    name: str
    seat: Seat
    group_id: Optional[str]
    baggage_size: BaggageSize
    status: BoardingStatus = field(default=BoardingStatus.WAITING)
    scheduled_slot: Optional[int] = field(default=None)
    board_at: Optional[datetime.datetime] = field(default=None)
    scanned_at: Optional[datetime.datetime] = field(default=None)


@dataclass
class PlaneConfig:
    name: str
    rows: int
    columns: tuple
    # Optional explicit col → SeatType mapping.  If None, position-based
    # logic is used (correct for symmetric narrowbody cabins).
    column_types: Optional[Dict[str, SeatType]] = field(default=None, repr=False)

    def seat_type_for(self, column: str) -> SeatType:
        """Return the SeatType for a given column letter."""
        col = column.upper()
        if self.column_types:
            return self.column_types.get(col, SeatType.AISLE)
        # Position-based fallback — works for symmetric narrowbody layouts.
        cols = [c.upper() for c in self.columns]
        idx = cols.index(col)
        n = len(cols)
        if idx == 0 or idx == n - 1:
            return SeatType.WINDOW
        if n <= 4:
            # 2-2 layout: no middle seats
            return SeatType.AISLE
        if idx == 1 or idx == n - 2:
            return SeatType.MIDDLE
        return SeatType.AISLE

    # ------------------------------------------------------------------
    # Narrowbody — 3+3
    # ------------------------------------------------------------------

    @classmethod
    def a320(cls) -> "PlaneConfig":
        """Airbus A320 — 30 rows, 180 seats, 3+3"""
        return cls("A320", 30, ("A", "B", "C", "D", "E", "F"))

    @classmethod
    def a321neo(cls) -> "PlaneConfig":
        """Airbus A321neo — 37 rows, 222 seats, 3+3"""
        return cls("A321neo", 37, ("A", "B", "C", "D", "E", "F"))

    @classmethod
    def b737(cls) -> "PlaneConfig":
        """Boeing 737-800 — 32 rows, 189 seats, 3+3"""
        return cls("B737-800", 32, ("A", "B", "C", "D", "E", "F"))

    @classmethod
    def b737max9(cls) -> "PlaneConfig":
        """Boeing 737 MAX 9 — 34 rows, 204 seats, 3+3"""
        return cls("B737 MAX 9", 34, ("A", "B", "C", "D", "E", "F"))

    # ------------------------------------------------------------------
    # Regional — 2+2
    # ------------------------------------------------------------------

    @classmethod
    def e195(cls) -> "PlaneConfig":
        """Embraer E195 — 24 rows, 96 seats, 2+2"""
        return cls("E195", 24, ("A", "B", "C", "D"))

    @classmethod
    def crj900(cls) -> "PlaneConfig":
        """Bombardier CRJ-900 — 23 rows, 90 seats, 2+2"""
        return cls("CRJ-900", 23, ("A", "B", "C", "D"))

    # ------------------------------------------------------------------
    # Widebody — 2+4+2  (A330)
    # ------------------------------------------------------------------

    @classmethod
    def a330_300(cls) -> "PlaneConfig":
        """Airbus A330-300 — 36 rows, 288 seats, 2+4+2"""
        return cls(
            "A330-300", 36,
            ("A", "B", "C", "D", "E", "F", "G", "H"),
            column_types={
                "A": SeatType.WINDOW, "B": SeatType.MIDDLE,   # left pair
                "C": SeatType.AISLE,  "D": SeatType.MIDDLE,   # centre-4
                "E": SeatType.MIDDLE, "F": SeatType.AISLE,    # centre-4
                "G": SeatType.MIDDLE, "H": SeatType.WINDOW,   # right pair
            },
        )

    # ------------------------------------------------------------------
    # Widebody — 3+3+3  (B787)
    # ------------------------------------------------------------------

    @classmethod
    def b787_9(cls) -> "PlaneConfig":
        """Boeing 787-9 — 36 rows, 296 seats, 3+3+3"""
        return cls(
            "B787-9", 36,
            ("A", "B", "C", "D", "E", "F", "G", "H", "K"),
            column_types={
                "A": SeatType.WINDOW, "B": SeatType.MIDDLE, "C": SeatType.AISLE,
                "D": SeatType.AISLE,  "E": SeatType.MIDDLE, "F": SeatType.AISLE,
                "G": SeatType.AISLE,  "H": SeatType.MIDDLE, "K": SeatType.WINDOW,
            },
        )

    # ------------------------------------------------------------------
    # Widebody — 3+4+3  (B777)
    # ------------------------------------------------------------------

    @classmethod
    def b777_300er(cls) -> "PlaneConfig":
        """Boeing 777-300ER — 42 rows, 420 seats, 3+4+3"""
        return cls(
            "B777-300ER", 42,
            ("A", "B", "C", "D", "E", "F", "G", "H", "J", "K"),
            column_types={
                "A": SeatType.WINDOW, "B": SeatType.MIDDLE, "C": SeatType.AISLE,
                "D": SeatType.AISLE,  "E": SeatType.MIDDLE, "F": SeatType.MIDDLE,
                "G": SeatType.AISLE,  "H": SeatType.AISLE,  "J": SeatType.MIDDLE,
                "K": SeatType.WINDOW,
            },
        )


@dataclass
class Flight:
    flight_number: str
    plane_config: PlaneConfig
    passengers: list  # List[Passenger]
    departure: Optional[datetime.datetime] = field(default=None)
