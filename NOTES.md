# Practical Boarding Planner — Project Notes

## What this is

A Python/Streamlit tool that implements the **Steffen (2008) MCMC-optimal boarding method**
for airline boarding.  It has three operating modes:

| Mode | Purpose |
|------|---------|
| **Planner** | Generate or upload a passenger manifest, view the Steffen boarding schedule, export CSV |
| **Gate Mode** | Manual gate tool — scan passengers, mark no-shows, handle late arrivals |
| **Simulation** | Automated sweep of no-show rates to evaluate boarding robustness |

---

## File layout

```
boarding_planner/
├── models.py       — dataclasses: Seat, Passenger, Flight, PlaneConfig (9 aircraft)
├── optimizer.py    — Steffen sequence: window → middle → aisle, back-to-front alternating
├── scheduler.py    — assigns board_at datetimes and slot durations
├── gate.py         — GateSession: scan / no-show / late-arrival with requeue
├── data_gen.py     — generate reproducible manifests; load from CSV
├── simulate.py     — programmatic boarding runs + sweep; SimResult dataclass
├── app.py          — Streamlit UI (4 modes + Help)
└── requirements.txt
```

---

## Key design decisions

### Boarding algorithm (optimizer.py)
- Phase order: **window → middle → aisle** (SeatType enum)
- Within each phase: alternating rows from the back (`_steffen_key`)
- Groups board as a unit placed in the phase of their **worst-positioned member**

### Late-passenger model (gate.py + simulate.py)
Realistic two-stage model (agreed in April 2026 session):

1. **`mark_no_show`** — removes passenger from the queue entirely, status = `NO_SHOW`.
   No automatic requeue.  Passenger is in limbo.
2. **`handle_late_arrival`** — called when the passenger physically returns to the gate.
   Calls `_requeue()` which inserts at the first slot whose avg phase ≥ passenger's phase,
   then rebuilds times.  Status back to `WAITING`.

Permanent no-shows never get `handle_late_arrival` called — they simply don't board.

### Simulation (simulate.py)
Each no-show is randomly classified as:
- **Permanent** (fraction = `permanent_noshow_frac`, default 10 %) — never returns
- **Late returner** — returns at `original_slot + randint(late_min_slots, late_max_slots)`

Parameters exposed as UI sliders in the Simulation expander.

`SimResult` fields of note:
- `late_arrived_count` — no-shows who returned in time and boarded
- `permanent_noshow_count` — permanent + returners who missed the door

---

## Bugs fixed (chronological)

| # | Symptom | Cause | Fix |
|---|---------|-------|-----|
| 1 | "Start Gate Mode" button did nothing | `key="app_mode"` on radio conflicted with button setting same key | Changed radio key to `mode_radio`; explicit `st.session_state.app_mode = mode` |
| 2 | `StreamlitAPIException` on mode switch | Button in `render_planner()` mutated session state after radio was bound | Added `_go_to_gate` boolean flag; pre-seed `mode_radio` before any widget renders |
| 3 | No-show passenger reappeared in the very next slot | `_requeue` searched from `_current_slot`, found the same slot they just missed | Changed to search from `_current_slot + 1`; later moved requeue entirely to `handle_late_arrival` |
| 4 | No-show passengers moved only one slot forward | `mark_no_show` was immediately requeueing — unrealistic | Split: `mark_no_show` = remove only; `handle_late_arrival` = requeue |
| 5 | Returning no-shows were re-marked as no-show | They were still in `noshow_ids` set when their new slot was processed | Added `requeued_ids` set; skip no-show check for passengers already requeued |
| 6 | Simulation appeared stuck / results disappeared | (a) `_try_advance` not called after all-noshow slots; (b) results stored in local var, wiped on every Streamlit rerender | (a) Added explicit `session._try_advance()` after each slot; (b) persist results in `st.session_state.sim_result` / `sweep_results` |

---

## Aircraft supported (PlaneConfig classmethods)

| Method | Name | Rows | Layout | Seats |
|--------|------|------|--------|-------|
| `a320()` | A320 | 30 | 3+3 | 180 |
| `a321neo()` | A321neo | 37 | 3+3 | 222 |
| `b737()` | B737-800 | 32 | 3+3 | 192 |
| `b737max9()` | B737 MAX 9 | 34 | 3+3 | 204 |
| `e195()` | E195 | 24 | 2+2 | 96 |
| `crj900()` | CRJ-900 | 23 | 2+2 | 92 |
| `a330_300()` | A330-300 | 36 | 2+4+2 | 288 |
| `b787_9()` | B787-9 | 36 | 3+3+3 | 324 |
| `b777_300er()` | B777-300ER | 42 | 3+4+3 | 420 |

---

## Run the app

```powershell
cd "~\boarding_planner"
~\AppData\Roaming\Python\Python312\Scripts\streamlit.exe run app.py
# opens at http://localhost:8501
```

---

## Potential next steps (not yet built)

- Export gate session events to CSV / PDF boarding report
- Visualise late-arrival timeline (when each returner re-entered vs boarding clock)
- Compare Steffen vs random vs back-to-front strategies in the Simulation sweep
- Sound / audio cue for the current boarding call in Gate Mode
