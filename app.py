"""
app.py — Streamlit boarding planner UI.

Two modes:
  Planner   — configure a flight, generate / upload a manifest, view the
              boarding schedule, download it as CSV.
  Gate Mode — simulate live boarding with per-passenger scan / no-show
              controls, automatic reoptimisation, and a live seat map.
"""
from __future__ import annotations

import base64
import datetime
import io
import math
import struct
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_gen import generate_flight_csv, load_passengers_from_df
from gate import GateSession
from models import BoardingStatus, Flight, PlaneConfig
from scheduler import build_schedule, SCHEDULE_EPOCH
from simulate import run_simulation, run_sweep, SimResult

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Boarding Planner",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_APP_MODES = ["Planner", "Gate Mode", "Simulation", "Help"]

_DEFAULTS: dict = {
    "manifest_df": None,
    "flight": None,
    "schedule_df": None,
    "gate_session": None,
    "app_mode": "Planner",
    "_go_to_gate": False,
    "_go_to_sim": False,
    "sim_result": None,
    "sweep_results": None,
    "gate_audio_enabled": False,
    "gate_audio_last_slot": -1,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Audio chime (computed once; cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def _chime_html() -> str:
    """Return an HTML <audio autoplay> snippet with a soft A5+E6 chime (PCM WAV)."""
    sr, dur = 8000, 0.35
    n = int(sr * dur)
    samples = [
        max(-32767, min(32767, int(
            math.exp(-6 * i / sr) * 32767
            * (0.5 * math.sin(2 * math.pi * 880 * i / sr)
               + 0.3 * math.sin(2 * math.pi * 1320 * i / sr))
        )))
        for i in range(n)
    ]
    data = struct.pack(f"<{n}h", *samples)
    hdr = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE",
        b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16,
        b"data", len(data),
    )
    b64 = base64.b64encode(hdr + data).decode()
    return (
        f"<audio autoplay style='display:none'>"
        f"<source src='data:audio/wav;base64,{b64}' type='audio/wav'>"
        f"</audio>"
    )


# Honour deferred mode switches — must run BEFORE any widget is rendered.
if st.session_state._go_to_gate:
    st.session_state.app_mode = "Gate Mode"
    st.session_state.mode_radio = "Gate Mode"
    st.session_state._go_to_gate = False
if st.session_state._go_to_sim:
    st.session_state.app_mode = "Simulation"
    st.session_state.mode_radio = "Simulation"
    st.session_state._go_to_sim = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
PLANE_OPTIONS = {
    # Narrowbody 3+3
    "A320  (30 rows · 180 seats · 3+3)": PlaneConfig.a320(),
    "A321neo (37 rows · 222 seats · 3+3)": PlaneConfig.a321neo(),
    "B737-800 (32 rows · 192 seats · 3+3)": PlaneConfig.b737(),
    "B737 MAX 9 (34 rows · 204 seats · 3+3)": PlaneConfig.b737max9(),
    # Regional 2+2
    "E195 (24 rows · 96 seats · 2+2)": PlaneConfig.e195(),
    "CRJ-900 (23 rows · 92 seats · 2+2)": PlaneConfig.crj900(),
    # Widebody
    "A330-300 (36 rows · 288 seats · 2+4+2)": PlaneConfig.a330_300(),
    "B787-9   (36 rows · 324 seats · 3+3+3)": PlaneConfig.b787_9(),
    "B777-300ER (42 rows · 420 seats · 3+4+3)": PlaneConfig.b777_300er(),
}

with st.sidebar:
    st.title("✈ Boarding Planner")

    mode: str = st.radio(
        "Mode",
        _APP_MODES,
        index=_APP_MODES.index(st.session_state.app_mode),
        key="mode_radio",
    )
    st.session_state.app_mode = mode

    st.divider()
    plane_label: str = st.selectbox("Aircraft", list(PLANE_OPTIONS.keys()))
    plane_config: PlaneConfig = PLANE_OPTIONS[plane_label]

    st.divider()
    if st.session_state.flight:
        n = len(st.session_state.flight.passengers)
        st.caption(f"Flight loaded: **{n}** passengers")
    else:
        st.caption("No flight loaded.")


# ---------------------------------------------------------------------------
# Seat map helper
# ---------------------------------------------------------------------------
def draw_seat_map(
    flight: Flight,
    gate_session: Optional[GateSession] = None,
) -> plt.Figure:
    """Render a colour-coded seat map.  Colours reflect boarding phase."""
    cfg = flight.plane_config
    cols = list(cfg.columns)
    rows = cfg.rows
    pax_map = {p.seat.seat_code: p for p in flight.passengers}

    PHASE_COLOR = {
        "window": "#4A90D9",
        "middle": "#F59B00",
        "aisle":  "#5CB85C",
    }
    BOARDED_COLOR = "#C8C8C8"
    NO_SHOW_COLOR = "#E74C3C"
    CURRENT_COLOR = "#FFD700"
    EMPTY_COLOR   = "#F0F0F0"

    fig_h = max(6.0, rows * 0.30)
    fig, ax = plt.subplots(figsize=(5.5, fig_h))
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("#F8F8F8")

    # IDs of passengers in the current call (highlighted gold)
    current_ids: set[str] = set()
    if gate_session:
        call = gate_session.get_current_call()
        if call:
            current_ids = {p.passenger_id for p in call}

    for r in range(1, rows + 1):
        y = rows - r  # row 1 at top = front of plane
        for c_idx, col in enumerate(cols):
            code = f"{r}{col}"
            p = pax_map.get(code)

            if p is None:
                color = EMPTY_COLOR
            elif gate_session and p.status == BoardingStatus.SCANNED:
                color = BOARDED_COLOR
            elif gate_session and p.status == BoardingStatus.NO_SHOW:
                color = NO_SHOW_COLOR
            elif gate_session and p.passenger_id in current_ids:
                color = CURRENT_COLOR
            else:
                color = PHASE_COLOR[p.seat.seat_type.value]

            rect = mpatches.FancyBboxPatch(
                (c_idx + 0.06, y + 0.06), 0.88, 0.88,
                boxstyle="round,pad=0.04",
                facecolor=color, edgecolor="#FFFFFF", linewidth=0.7,
            )
            ax.add_patch(rect)

            # Small dot marks group members
            if p and p.group_id:
                ax.plot(c_idx + 0.5, y + 0.5, "k.", markersize=2.5, alpha=0.55)

    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, rows)
    ax.set_xticks([i + 0.5 for i in range(len(cols))])
    ax.set_xticklabels(list(cols), fontsize=8)
    ax.set_yticks([rows - r + 0.5 for r in range(1, rows + 1)])
    ax.set_yticklabels([str(r) for r in range(1, rows + 1)], fontsize=6)
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_items = [
        mpatches.Patch(color="#4A90D9", label="Window phase"),
        mpatches.Patch(color="#F59B00", label="Middle phase"),
        mpatches.Patch(color="#5CB85C", label="Aisle phase"),
    ]
    if gate_session:
        legend_items += [
            mpatches.Patch(color="#FFD700", label="Current call"),
            mpatches.Patch(color="#C8C8C8", label="Boarded"),
            mpatches.Patch(color="#E74C3C", label="No-show (requeued)"),
        ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=7, framealpha=0.9)
    ax.set_title("Seat map — boarding phases", fontsize=9, pad=4)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Planner mode
# ---------------------------------------------------------------------------
def render_planner() -> None:
    st.header("Flight Setup & Boarding Schedule")
    tab_setup, tab_schedule = st.tabs(["✈  Flight Setup", "📋  Schedule"])

    # ---- Setup tab -------------------------------------------------------
    with tab_setup:
        col_gen, col_up = st.columns(2, gap="large")

        with col_gen:
            st.subheader("Generate random flight")
            group_ratio = st.slider(
                "Group passenger ratio", 0.0, 0.60, 0.30, 0.05,
                help="Fraction of seats assigned to travel groups of 2–4.",
            )
            seed = st.number_input(
                "Random seed", value=42, step=1,
                help="Fixed seed → fully reproducible manifest.",
            )
            if st.button("Generate", type="primary"):
                df = generate_flight_csv(
                    plane_config=plane_config,
                    group_ratio=group_ratio,
                    seed=int(seed),
                )
                _load_flight(df, "BPL001")

        with col_up:
            st.subheader("Upload manifest CSV")
            st.caption(
                "Required columns: `passenger_id`, `name`, `row`, `column`, "
                "`group_id`, `baggage_size`"
            )
            uploaded = st.file_uploader("Choose CSV file", type="csv")
            if uploaded:
                if st.button("Load CSV", type="primary"):
                    try:
                        df = pd.read_csv(uploaded)
                        _load_flight(df, "UPLOAD")
                    except Exception as exc:
                        st.error(f"Could not parse CSV: {exc}")

        if st.session_state.manifest_df is not None:
            st.divider()
            st.subheader("Manifest preview")
            st.dataframe(
                st.session_state.manifest_df,
                width="stretch",
                height=280,
            )

            st.divider()
            st.markdown("#### ✅ Flight ready — what would you like to do next?")
            _nc1, _nc2, _nc3 = st.columns(3, gap="large")

            with _nc1:
                st.markdown("**📋 Review Schedule**")
                st.caption(
                    "Open the **Schedule** tab above to inspect the full "
                    "boarding sequence, see the phase seat map, and download the CSV."
                )

            with _nc2:
                st.markdown("**🚪 Gate Mode** — Manual boarding")
                st.caption(
                    "Scan passengers through one by one, mark no-shows, handle "
                    "late arrivals, and watch the live seat map update in real time."
                )
                if st.button("🚪 Start Gate Mode", type="primary", key="setup_gate_btn"):
                    for p in st.session_state.flight.passengers:
                        p.status = BoardingStatus.WAITING
                        p.scanned_at = None
                    st.session_state.gate_session = GateSession(
                        flight=st.session_state.flight,
                        boarding_start=datetime.datetime.now(),
                    )
                    st.session_state._go_to_gate = True
                    st.rerun()

            with _nc3:
                st.markdown("**📊 Simulation** — Automated testing")
                st.caption(
                    "Stress-test the boarding plan with configurable no-show rates "
                    "and measure smoothness scores without manual input."
                )
                if st.button("📊 Run Simulation", key="setup_sim_btn"):
                    st.session_state._go_to_sim = True
                    st.rerun()

    # ---- Schedule tab ----------------------------------------------------
    with tab_schedule:
        if st.session_state.schedule_df is None or st.session_state.flight is None:
            st.info("Generate or upload a flight in the **Flight Setup** tab first.")
            return

        col_l, col_r = st.columns([1, 1], gap="large")

        with col_l:
            st.subheader("Boarding schedule")
            df: pd.DataFrame = st.session_state.schedule_df
            st.dataframe(df, width="stretch", height=480)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇  Download schedule CSV",
                data=csv_bytes,
                file_name="boarding_schedule.csv",
                mime="text/csv",
            )

            if len(df) > 0:
                first_t = df["board_at"].iloc[0]
                last_t  = df["board_at"].iloc[-1]
                total_slots = df["slot"].nunique()
                st.caption(
                    f"**{total_slots}** slots · window: {first_t} → {last_t}"
                )

        with col_r:
            st.subheader("Seat map")
            fig = draw_seat_map(st.session_state.flight)
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        st.divider()
        _sched_btn1, _sched_btn2 = st.columns(2, gap="medium")
        with _sched_btn1:
            if st.button("🚪  Start Gate Mode", type="primary", key="sched_gate_btn"):
                # Reset all passenger statuses for a clean session
                for p in st.session_state.flight.passengers:
                    p.status = BoardingStatus.WAITING
                    p.scanned_at = None
                st.session_state.gate_session = GateSession(
                    flight=st.session_state.flight,
                    boarding_start=datetime.datetime.now(),
                )
                st.session_state._go_to_gate = True
                st.rerun()
        with _sched_btn2:
            if st.button("📊  Run Simulation", key="sched_sim_btn"):
                st.session_state._go_to_sim = True
                st.rerun()


def _load_flight(df: pd.DataFrame, flight_number: str) -> None:
    """Parse a manifest DataFrame, build a Flight, and compute the schedule."""
    passengers = load_passengers_from_df(df, plane_config)
    flight = Flight(
        flight_number=flight_number,
        plane_config=plane_config,
        passengers=passengers,
        departure=None,
    )
    st.session_state.manifest_df = df
    st.session_state.flight = flight
    st.session_state.schedule_df = build_schedule(flight, SCHEDULE_EPOCH)
    st.session_state.gate_session = None
    st.session_state.sim_result = None
    st.session_state.sweep_results = None
    st.success(f"Loaded **{len(passengers)}** passengers.")


# ---------------------------------------------------------------------------
# Gate mode
# ---------------------------------------------------------------------------
def render_gate() -> None:
    session: Optional[GateSession] = st.session_state.gate_session
    if session is None:
        st.warning(
            "No active gate session. "
            "Go to **Planner → Flight Setup**, generate or upload a flight, "
            "then click **🚪 Start Gate Mode**."
        )
        return

    stats = session.stats()

    # --- Stats bar ---------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Boarded",     f"{stats['boarded']} / {stats['total']}")
    c2.metric("Remaining",   stats["remaining"])
    c3.metric("No-show",     stats["no_show"])
    c4.metric("Compliance",  f"{stats['compliance_pct']} %")
    elapsed = stats["elapsed_seconds"]
    c5.metric("Elapsed",     f"{elapsed // 60}m {elapsed % 60:02d}s")

    st.divider()

    if session.is_complete:
        st.success("✅  Boarding complete!")
        st.balloons()
        _render_summary(session)
        return

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        _render_current_call(session)
        st.divider()
        _render_upcoming(session)

    with col_right:
        _render_event_log(session)
        if _no_shows(session):
            st.divider()
            _render_late_arrivals(session)

    # Live seat map
    with st.expander("🗺  Seat map", expanded=True):
        _tog_col, _ = st.columns([1, 5])
        with _tog_col:
            st.toggle(
                "🔊 Audio cues",
                key="gate_audio_enabled",
                help="Play a soft chime whenever the boarding call advances to the next slot.",
            )
        fig = draw_seat_map(st.session_state.flight, gate_session=session)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    # Inject chime when the slot index has just advanced
    if st.session_state.gate_audio_enabled and not session.is_complete:
        _cur = session.current_slot_index
        if st.session_state.gate_audio_last_slot != _cur:
            st.session_state.gate_audio_last_slot = _cur
            st.markdown(_chime_html(), unsafe_allow_html=True)


def _render_current_call(session: GateSession) -> None:
    slot_idx = session.current_slot_index
    total    = session.total_slots
    st.subheader(f"Now boarding  —  slot {slot_idx + 1} / {total}")

    current = session.get_current_call()
    if not current:
        st.info("All slots complete.")
        return

    waiting = [p for p in current if p.status != BoardingStatus.SCANNED]

    for p in waiting:
        group_tag = f"Group **{p.group_id}**" if p.group_id else "Solo"
        st.markdown(
            f"**{p.name}** &nbsp;·&nbsp; `{p.seat.seat_code}` "
            f"({p.seat.seat_type.value}) &nbsp;·&nbsp; {group_tag}  \n"
            f"Baggage: *{p.baggage_size.value}*"
        )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("✓ Scan", key=f"scan_{p.passenger_id}", type="primary"):
                session.scan_passenger(p.passenger_id)
                st.rerun()
        with b2:
            if st.button("✗ No-show", key=f"noshow_{p.passenger_id}"):
                session.mark_no_show(p.passenger_id)
                st.rerun()
        st.divider()

    if len(waiting) > 1:
        if st.button("✓  Scan all in this slot", type="secondary"):
            for p in waiting:
                session.scan_passenger(p.passenger_id)
            st.rerun()


def _render_upcoming(session: GateSession) -> None:
    st.subheader("Upcoming slots")
    upcoming = session.get_upcoming_slots(4)
    if not upcoming:
        st.caption("No more slots.")
        return
    for i, slot in enumerate(upcoming, start=1):
        names = ", ".join(p.name for p in slot)
        seats = ", ".join(p.seat.seat_code for p in slot)
        board_at = slot[0].board_at
        if board_at:
            _e = int((board_at - session.boarding_start).total_seconds())
            time_str = f"+{_e // 60}m {_e % 60:02d}s"
        else:
            time_str = "—"
        st.caption(f"**+{i}** &nbsp; {time_str} &nbsp; {names} — `{seats}`")


def _render_event_log(session: GateSession) -> None:
    st.subheader("Event log")
    events = list(reversed(session.events))
    if not events:
        st.caption("No events yet.")
        return

    ICONS = {
        "scanned":      "✅",
        "no_show":      "⚠️",
        "late_arrival": "🔔",
    }
    for ev in events[:25]:
        icon = ICONS.get(ev.event_type, "•")
        ts   = ev.timestamp.strftime("%H:%M:%S")
        st.caption(f"{icon} `{ts}` {ev.message}")


def _render_late_arrivals(session: GateSession) -> None:
    st.subheader("Late arrivals")
    for p in _no_shows(session):
        if p.board_at:
            _e = int((p.board_at - session.boarding_start).total_seconds())
            board_at_str = f"+{_e // 60}m {_e % 60:02d}s"
        else:
            board_at_str = "—"
        col_a, col_b = st.columns([3, 1])
        col_a.caption(
            f"**{p.name}** — `{p.seat.seat_code}` · "
            f"slot {p.scheduled_slot} ({board_at_str})"
        )
        with col_b:
            if st.button("🔔 Arrived", key=f"late_{p.passenger_id}"):
                session.handle_late_arrival(p.passenger_id)
                st.rerun()


def _render_summary(session: GateSession) -> None:
    stats = session.stats()
    st.subheader("Boarding summary")
    c1, c2, c3 = st.columns(3)
    elapsed = stats["elapsed_seconds"]
    c1.metric("Total boarded",    f"{stats['boarded']} / {stats['total']}")
    c2.metric("Final compliance", f"{stats['compliance_pct']} %")
    c3.metric("Total time",       f"{elapsed // 60}m {elapsed % 60:02d}s")
    remaining_ns = len([p for p in session.flight.passengers
                        if p.status == BoardingStatus.NO_SHOW])
    if remaining_ns:
        st.warning(f"{remaining_ns} passenger(s) did not board.")


def _no_shows(session: GateSession):
    return [p for p in session.flight.passengers if p.status == BoardingStatus.NO_SHOW]


# ---------------------------------------------------------------------------
# Boarding animation helpers
# ---------------------------------------------------------------------------

# Aisle gap (in seat-width units) inserted before these column indices.
_AISLE_BEFORE: dict[int, dict[int, float]] = {
    4:  {2: 0.8},                  # 2+2
    6:  {3: 0.8},                  # 3+3
    8:  {2: 0.8, 6: 0.8},          # 2+4+2
    9:  {3: 0.8, 6: 0.8},          # 3+3+3
    10: {3: 0.8, 7: 0.8},          # 3+4+3
}


def _column_x_positions(n_cols: int) -> list:
    """Return x-axis positions for seat columns, inserting aisle gaps."""
    gaps = _AISLE_BEFORE.get(n_cols, {})
    x, positions = 0.0, []
    for i in range(n_cols):
        x += gaps.get(i, 0.0)
        positions.append(x)
        x += 1.0
    return positions


def _make_boarding_animation(result: SimResult) -> go.Figure:
    """
    Build a Plotly animated seat-map figure that replays the simulation result.
    Seats light up in phase colour as each passenger boards; no-shows stay red.
    Animation plays entirely client-side — no Streamlit reruns needed.
    """
    flight = st.session_state.flight
    cfg    = flight.plane_config
    cols   = list(cfg.columns)
    n_cols = len(cols)
    n_rows = cfg.rows
    col_x  = _column_x_positions(n_cols)

    boarded_codes: set  = {r.seat_code for r in result.records}
    board_time_of: dict = {r.seat_code: r.actual_board_at for r in result.records}
    seat_type_of: dict  = {
        p.seat.seat_code: p.seat.seat_type.value for p in flight.passengers
    }

    PHASE_COLOR = {"window": "#4A90D9", "middle": "#F59B00", "aisle": "#5CB85C"}
    EMPTY  = "#E8E8E8"
    NOSHOW = "#E74C3C"

    # Build per-seat position arrays (fixed for all frames)
    xs, ys, texts, symbols = [], [], [], []
    seat_order: list = []
    for p in flight.passengers:
        col_upper = p.seat.column.upper()
        if col_upper not in cols:
            continue
        ci = cols.index(col_upper)
        xs.append(col_x[ci])
        ys.append(n_rows + 1 - p.seat.row)   # row 1 at top
        texts.append(f"{p.name}<br>{p.seat.seat_code}")
        symbols.append("square" if p.seat.seat_code in boarded_codes else "x")
        seat_order.append(p.seat.seat_code)

    # Time range for animation
    all_times = sorted(board_time_of.values())
    if not all_times:
        return go.Figure()
    t_start  = all_times[0]
    total_s  = max(1, (all_times[-1] - t_start).total_seconds())
    N_FRAMES = 50
    time_steps = [
        t_start + datetime.timedelta(seconds=total_s * i / (N_FRAMES - 1))
        for i in range(N_FRAMES)
    ]

    def _colors_at(t: datetime.datetime) -> list:
        out = []
        for code in seat_order:
            if code not in boarded_codes:
                out.append(NOSHOW)
            elif board_time_of[code] <= t:
                out.append(PHASE_COLOR[seat_type_of[code]])
            else:
                out.append(EMPTY)
        return out

    def _trace(colors: list) -> go.Scatter:
        return go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(
                symbol=symbols,
                color=colors,
                size=9,
                line=dict(width=0.5, color="#FFFFFF"),
            ),
            text=texts,
            hoverinfo="text",
            showlegend=False,
        )

    # Animation frames
    frames = [
        go.Frame(
            data=[_trace(_colors_at(t))],
            name=str(i),
            layout=go.Layout(title=dict(
                text=f"Elapsed: {int((t - t_start).total_seconds()) // 60}m "
                     f"{int((t - t_start).total_seconds()) % 60:02d}s"
            )),
        )
        for i, t in enumerate(time_steps)
    ]

    slider_steps = [
        {
            "args": [[f.name], {
                "frame": {"duration": 0, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 0},
            }],
            "label": "",
            "method": "animate",
        }
        for f in frames
    ]

    fig = go.Figure(
        data=[_trace(_colors_at(time_steps[0]))],
        frames=frames,
    )

    # Static legend traces (invisible points, legend only)
    for color, label in [
        ("#4A90D9", "Window"),
        ("#F59B00", "Middle"),
        ("#5CB85C", "Aisle"),
        ("#E74C3C", "No-show"),
        ("#E8E8E8", "Not yet boarded"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=color, symbol="square", size=9),
            name=label, showlegend=True,
        ))

    fig.update_layout(
        height=max(420, n_rows * 13),
        title=dict(text="Elapsed: 0m 00s", x=0.5),
        xaxis=dict(
            tickvals=col_x, ticktext=cols,
            title="Column",
            range=[-0.6, col_x[-1] + 0.6],
            zeroline=False, showgrid=False,
        ),
        yaxis=dict(
            tickvals=[n_rows + 1 - r for r in
                      range(1, n_rows + 1, max(1, n_rows // 10))],
            ticktext=[str(r) for r in
                      range(1, n_rows + 1, max(1, n_rows // 10))],
            title="Row", range=[0, n_rows + 2],
            zeroline=False, showgrid=False,
        ),
        plot_bgcolor="#F8F8F8",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=60, r=20, t=80, b=70),
        legend=dict(x=1.02, y=0.95, font=dict(size=10)),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0, "xanchor": "left",
            "y": 1.10, "yanchor": "top",
            "bgcolor": "#1A73E8",
            "bordercolor": "#1558B0",
            "font": {"size": 13, "color": "#FFFFFF", "family": "Arial, sans-serif"},
            "buttons": [
                {
                    "label": "▶  Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 180, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    }],
                },
                {
                    "label": "⏸  Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                },
            ],
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top", "xanchor": "left",
            "currentvalue": {"visible": False},
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.9, "x": 0.1, "y": 0,
            "steps": slider_steps,
        }],
    )
    return fig


# ---------------------------------------------------------------------------
# Simulation mode
# ---------------------------------------------------------------------------

def render_simulation() -> None:
    st.header("Boarding Robustness Simulation")

    if st.session_state.flight is None:
        st.info("Generate or upload a flight in **Planner → Flight Setup** first.")
        return

    flight = st.session_state.flight

    tab_single, tab_sweep = st.tabs(["Single scenario", "Sweep (0 – 80 %)"])

    # Shared late-arrival parameters (shown above both tabs)
    with st.expander("⚙️  Late-arrival model", expanded=False):
        st.caption(
            "Controls how no-show passengers behave after missing their slot. "
            "This mirrors real life: some passengers wander back, others never return."
        )
        la_col1, la_col2, la_col3 = st.columns(3)
        perm_frac = la_col1.slider(
            "Permanent no-shows", 0, 50, 10, step=5, format="%d %%",
            help="Fraction of no-shows who never return to the gate.",
            key="perm_frac",
        )
        late_min = la_col2.slider(
            "Min return delay (slots)", 1, 20, 5, step=1,
            help="Earliest a no-show can return, in boarding slots after their original slot.",
            key="late_min",
        )
        late_max = la_col3.slider(
            "Max return delay (slots)", 5, 60, 25, step=5,
            help="Latest a no-show can return. Those who haven't returned by boarding end are treated as permanent.",
            key="late_max",
        )

    # ── Single scenario ──────────────────────────────────────────────────
    with tab_single:
        col_l, col_r = st.columns([1, 2], gap="large")
        with col_l:
            single_rate = st.slider(
                "No-show rate", 0, 80, 20, step=5,
                format="%d %%",
                help="Fraction of passengers who miss their assigned slot.",
            )
            single_seed = st.number_input("Seed", value=42, step=1, key="single_seed")
            run_single = st.button("▶  Run simulation", type="primary")

        if run_single:
            late_min_eff = min(int(late_min), int(late_max))
            late_max_eff = max(int(late_min), int(late_max))
            with st.spinner("Simulating…"):
                st.session_state.sim_result = run_simulation(
                    flight,
                    no_show_rate=single_rate / 100,
                    seed=int(single_seed),
                    permanent_noshow_frac=perm_frac / 100,
                    late_min_slots=late_min_eff,
                    late_max_slots=late_max_eff,
                )

        result: Optional[SimResult] = st.session_state.sim_result
        if result is not None:
            with col_r:
                _render_single_result(result)

            st.divider()
            # --- Optional animated boarding visualisation -----------------
            show_viz = st.checkbox(
                "📊  Show animated boarding visualisation",
                value=False,
                key="show_boarding_viz",
                help=(
                    "Animated seat map: seats light up as passengers board. "
                    "Plays client-side — use ▶ Play or drag the timeline slider."
                ),
            )
            if show_viz:
                with st.spinner("Building animation…"):
                    anim_fig = _make_boarding_animation(result)
                st.plotly_chart(anim_fig, width="stretch")
                st.caption(
                    "🔴 **Red ✕ seats** are passengers who never boarded — "
                    "either permanent no-shows who didn't return to the gate, "
                    "or late returners who missed the final boarding call."
                )

            st.divider()
            st.subheader("Passenger detail")
            df_rec = _records_to_df(result)
            st.dataframe(
                df_rec.style.apply(_highlight_noshow, axis=1),
                width="stretch",
                height=380,
            )
            st.download_button(
                "⬇  Download detail CSV",
                data=df_rec.to_csv(index=False).encode(),
                file_name=f"sim_{single_rate}pct.csv",
                mime="text/csv",
            )

    # ── Sweep ─────────────────────────────────────────────────────────────
    with tab_sweep:
        col_l2, col_r2 = st.columns([1, 2], gap="large")
        with col_l2:
            sweep_seed = st.number_input("Base seed", value=42, step=1, key="sweep_seed")
            step_size  = st.selectbox("Step size", [5, 10], index=0, key="sweep_step")
            run_sweep_btn = st.button("▶  Run sweep", type="primary", key="run_sweep")

        if run_sweep_btn:
            rates = [r / 100 for r in range(0, 85, step_size)]
            late_min_eff = min(int(late_min), int(late_max))
            late_max_eff = max(int(late_min), int(late_max))
            with st.spinner(f"Running {len(rates)} scenarios…"):
                st.session_state.sweep_results = run_sweep(
                    flight, rates=rates, seed=int(sweep_seed),
                    permanent_noshow_frac=perm_frac / 100,
                    late_min_slots=late_min_eff,
                    late_max_slots=late_max_eff,
                )

        results = st.session_state.sweep_results
        if results is not None:
            with col_r2:
                _render_sweep_charts(results)

            st.divider()
            sweep_df = _sweep_to_df(results)
            st.dataframe(sweep_df, width="stretch")
            st.download_button(
                "⬇  Download sweep CSV",
                data=sweep_df.to_csv(index=False).encode(),
                file_name="sweep_results.csv",
                mime="text/csv",
            )


def _render_single_result(r: SimResult) -> None:
    c1, c2, c3, c4 = st.columns(4)
    def _fmt(s: int) -> str:
        return f"{s // 60}m {s % 60:02d}s"

    c1.metric("Boarding time",   _fmt(r.total_seconds),
              delta=f"+{_fmt(r.total_seconds - r.baseline_seconds)} vs baseline",
              delta_color="inverse")
    c2.metric("Time overhead",   f"{r.time_overhead_pct} %",  delta_color="inverse")
    c3.metric("Avg drift / pax", f"{r.avg_drift_seconds:.0f}s")
    c4.metric("Smoothness score", f"{r.smoothness_score} / 100",
              delta_color="normal")

    # Second row: late-arrival breakdown
    ca, cb, cc = st.columns(3)
    ca.metric("Missed slot",      r.no_show_count)
    cb.metric("Returned & boarded", r.late_arrived_count,
              help="No-shows who came back in time and were requeued.")
    cc.metric("Did not board",    r.permanent_noshow_count,
              delta_color="inverse",
              help="Permanent no-shows + late returners who missed the door.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    # Drift histogram
    drifts = [
        max(0.0, (rec.actual_board_at - rec.planned_board_at).total_seconds())
        for rec in r.records
    ]
    axes[0].hist(drifts, bins=20, color="#4A90D9", edgecolor="white")
    axes[0].set_xlabel("Drift (seconds)")
    axes[0].set_ylabel("Passengers")
    axes[0].set_title("Schedule drift distribution")

    # Pie: on-time vs returned late vs permanent no-show
    ns_returned = r.late_arrived_count
    ns_permanent = r.permanent_noshow_count
    on_time = r.total_passengers - r.no_show_count
    slices = [on_time, ns_returned, ns_permanent]
    labels = ["On time", "Late (returned)", "Did not board"]
    colors = ["#5CB85C", "#F59B00", "#E74C3C"]
    # Drop zero slices to avoid matplotlib warnings
    filtered = [(s, l, c) for s, l, c in zip(slices, labels, colors) if s > 0]
    axes[1].pie(
        [f[0] for f in filtered],
        labels=[f[1] for f in filtered],
        colors=[f[2] for f in filtered],
        autopct="%1.0f%%",
        startangle=90,
    )
    axes[1].set_title("Slot compliance")

    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _render_sweep_charts(results: list) -> None:
    rates   = [r.no_show_rate * 100 for r in results]
    times   = [r.total_seconds / 60  for r in results]
    scores  = [r.smoothness_score    for r in results]
    drifts  = [r.avg_drift_seconds   for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    axes[0].plot(rates, times, "o-", color="#4A90D9", linewidth=2)
    axes[0].axhline(times[0], linestyle="--", color="#999", linewidth=1, label="baseline")
    axes[0].set_xlabel("No-show rate (%)")
    axes[0].set_ylabel("Boarding time (min)")
    axes[0].set_title("Total boarding time")
    axes[0].legend(fontsize=8)

    axes[1].plot(rates, scores, "o-", color="#5CB85C", linewidth=2)
    axes[1].set_ylim(0, 105)
    axes[1].axhline(80, linestyle="--", color="#F59B00", linewidth=1, label="80 pt threshold")
    axes[1].set_xlabel("No-show rate (%)")
    axes[1].set_ylabel("Score (0–100)")
    axes[1].set_title("Smoothness score")
    axes[1].legend(fontsize=8)

    axes[2].plot(rates, drifts, "o-", color="#E74C3C", linewidth=2)
    axes[2].set_xlabel("No-show rate (%)")
    axes[2].set_ylabel("Seconds")
    axes[2].set_title("Avg drift per passenger")

    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # Find tipping point: first rate where score < 80
    tip = next((r for r in results if r.smoothness_score < 80), None)
    if tip:
        st.info(
            f"Smoothness drops below 80 at **{tip.no_show_rate * 100:.0f}% no-show rate** "
            f"(score: {tip.smoothness_score}, overhead: +{tip.time_overhead_pct}%)."
        )
    else:
        st.success("Smoothness stays above 80 across the entire sweep — the requeue strategy is very robust.")


def _records_to_df(result: SimResult) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "passenger_id": r.passenger_id,
            "name":         r.name,
            "seat":         r.seat_code,
            "seat_type":    r.seat_type,
            "group_id":     r.group_id or "—",
            "no_show":      r.was_noshow,
            "planned_slot": r.planned_slot,
            "actual_slot":  r.actual_slot,
            "slot_delta":   r.actual_slot - r.planned_slot,
            "drift_s":      max(0, int((r.actual_board_at - r.planned_board_at).total_seconds())),
        }
        for r in result.records
    ])


def _sweep_to_df(results: list) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "no_show_pct":         f"{r.no_show_rate * 100:.0f}%",
            "no_show_count":       r.no_show_count,
            "late_returned":       r.late_arrived_count,
            "permanent_noboard":   r.permanent_noshow_count,
            "boarding_time":       f"{r.total_seconds // 60}m {r.total_seconds % 60:02d}s",
            "time_overhead_pct":   f"{r.time_overhead_pct}%",
            "avg_drift_s":         r.avg_drift_seconds,
            "phase_violations":    r.phase_violations,
            "smoothness_score":    r.smoothness_score,
        }
        for r in results
    ])


def _highlight_noshow(row: pd.Series) -> list:
    color = "background-color: #fde8e8" if row.get("no_show") else ""
    return [color] * len(row)


# ---------------------------------------------------------------------------
# Help / About
# ---------------------------------------------------------------------------

def render_help() -> None:
    st.header("✈  Boarding Planner — Help & Background")

    st.markdown("""
    This tool implements and evaluates **Steffen-method boarding** — an
    algorithmically optimal passenger sequence that minimises aisle interference
    by letting multiple passengers stow baggage simultaneously.  
    It is built as a practical gate planning tool with three operating modes:
    **Planner**, **Gate Mode**, and **Simulation**.
    """)

    # ------------------------------------------------------------------
    tab_how, tab_modes, tab_metrics, tab_refs = st.tabs([
        "How it works",
        "Using the modes",
        "Metrics explained",
        "References & further reading",
    ])

    # ── How it works ──────────────────────────────────────────────────
    with tab_how:
        st.subheader("The Steffen Method")
        st.markdown("""
        Jason H. Steffen (2008) applied a **Markov Chain Monte Carlo (MCMC)**
        optimisation algorithm — borrowed from statistical physics — to find the
        passenger boarding sequence that minimises total boarding time.

        The algorithm converges on a clear pattern:

        | Phase | Who boards | Why |
        |-------|-----------|-----|
        | 1 | **Window seats**, every other row (even offset from back) | Maximum parallel bag-stowing — passengers are far enough apart to work simultaneously |
        | 2 | **Window seats**, remaining rows | Fills the gaps |
        | 3 | **Middle seats**, same alternating pattern | Same logic |
        | 4 | **Aisle seats** last | They board fastest; keeping the aisle clear matters most early on |

        The key insight: **aisle interference** — one passenger blocking the queue
        while stowing — is the dominant cost.  Steffen eliminates it by ensuring
        no two passengers in the same boarding batch are close enough to block
        each other.
        """)

        st.subheader("How groups are handled")
        st.markdown("""
        Travel groups (families, couples) must board together.  This app uses
        **Option A (simple group slotting)**:

        - The group's boarding phase is determined by its **worst-positioned
          member** — the one who would board latest if travelling alone.
        - The entire group boards as a single unit in that slot.
        - This is suboptimal compared to splitting groups, but operationally
          realistic and socially acceptable.
        """)

        st.subheader("Late-passenger requeue strategy")
        st.markdown("""
        When a passenger misses their slot, the system **silently requeues** them
        — no gate agent action required:

        1. If their **boarding phase is still active** (e.g. window seats are
           still being called), they are inserted at the earliest remaining
           slot in that phase.  The parallelism benefit is partially preserved.
        2. If their phase has **already passed**, they are inserted at the
           next open slot — boarding as soon as possible without disrupting
           the remaining queue.

        This strategy was chosen to maximise flow smoothness: it avoids the
        queue gap that would occur if a missed slot were simply left empty,
        and avoids the phase violation that would occur if they were appended
        at the very end.
        """)

    # ── Using the modes ───────────────────────────────────────────────
    with tab_modes:
        st.subheader("1  ·  Planner")
        st.markdown("""
        **Purpose:** Create a boarding schedule for a flight.

        **Steps:**
        1. Select an **aircraft type** in the sidebar.
        2. In the **Flight Setup** tab, either:
           - Click **Generate** to create a random reproducible manifest
             (adjust group ratio and seed), or
           - Upload your own **CSV** with columns:
             `passenger_id`, `name`, `row`, `column`, `group_id`, `baggage_size`
             *(baggage_size: `none` / `small` / `large`)*
        3. Switch to the **Schedule** tab — the full boarding sequence,
           time slots, and a colour-coded seat map are displayed.
        4. Download the schedule as CSV for printing or distribution.
        5. Click **🚪 Start Gate Mode** to begin live boarding.
        """)

        st.subheader("2  ·  Gate Mode")
        st.markdown("""
        **Purpose:** Simulate live boarding at the gate with real-time feedback.

        **Steps:**
        - The **Now boarding** panel shows the current slot's passengers.
        - For each passenger click **✓ Scan** (boarded) or **✗ No-show** (missed slot).
        - **Scan all** processes the whole slot at once when no complications.
        - No-show passengers appear in the **Late arrivals** panel; when they
          show up, click **🔔 Arrived** — they are already requeued automatically.
        - **Upcoming slots** gives the gate agent a look-ahead of the next 4 slots.
        - The **event log** records every action with timestamps.
        - The **seat map** updates live: gold = current call, grey = boarded,
          red = no-show requeued.
        - Enable **🔊 Audio cues** (inside the seat map expander) to hear a soft
          chime whenever the boarding call advances to the next slot.
        - When all slots are complete, a summary screen is shown.
        """)

        st.subheader("3  ·  Simulation")
        st.markdown("""
        **Purpose:** Evaluate how robust the boarding plan is under different
        levels of passenger non-compliance.

        **Single scenario tab:**
        - Set a **no-show rate** (fraction of passengers who miss their slot).
        - Run the simulation to see boarding time, time overhead vs baseline,
          average schedule drift, smoothness score, and per-passenger detail.
        - Tick **📊 Show animated boarding visualisation** to replay the
          simulated boarding frame-by-frame on an interactive seat map.

        **Sweep tab:**
        - Runs the simulation for every rate from 0 % to 80 % in configurable
          steps.
        - Produces three charts: boarding time, smoothness score, and average drift.
        - Automatically identifies the **tipping point** — the no-show rate at
          which smoothness drops below 80/100.
        - Download sweep results as CSV for offline analysis.
        """)

    # ── Metrics ───────────────────────────────────────────────────────
    with tab_metrics:
        st.subheader("Metric definitions")
        st.markdown("""
        | Metric | Definition |
        |--------|------------|
        | **Boarding time** | Total simulated seconds from first call to last scan |
        | **Baseline time** | Boarding time with 0 % no-shows (perfect compliance) |
        | **Time overhead %** | `(boarding_time − baseline) / baseline × 100` |
        | **Avg drift / pax** | Mean seconds each passenger boarded *later* than their original planned slot.  Measures how far individual schedules slipped. |
        | **Phase violations** | Passengers who ultimately boarded in a *later* phase than originally assigned (e.g. a window-seat passenger boarding during the aisle phase).  Indicates breakdown of parallelism. |
        | **Smoothness score** | Composite 0–100 score: `100 − (overhead_penalty × 0.5) − drift_penalty`.  Overhead is capped at 100 %; drift penalty maxes at 50 points (≈ 6 min avg drift). |

        **Interpreting the smoothness score:**

        | Score | Interpretation |
        |-------|----------------|
        | 90–100 | Excellent — plan is robust; minor disruptions fully absorbed |
        | 70–89  | Good — noticeable overhead but flow is maintained |
        | 50–69  | Moderate — meaningful delay; passengers experience queuing |
        | < 50   | Poor — plan has broken down; random boarding would perform similarly |

        **Why 10 % non-compliance matters so much:**  
        Steffen's own paper (2008) showed that even 10 % random swaps
        significantly increase mean boarding time.  Our simulation confirms
        this — for a 180-seat A320 the smoothness score falls from 100 → ~57
        at 10 % non-compliance.  This is the core practical limitation of
        any highly-ordered boarding strategy.
        """)

    # ── References ────────────────────────────────────────────────────
    with tab_refs:
        st.subheader("Original research")
        st.markdown("""
        - **Steffen, J.H. (2008)**  
          *Optimal Boarding Method for Airline Passengers*  
          [arXiv:0802.0733](https://arxiv.org/abs/0802.0733)  
          The founding paper. MCMC optimisation on a simulated 757. Introduces
          the alternating window-first sequence and quantifies aisle interference.

        - **Steffen, J.H. & Hotchkiss, J. (2012)**  
          *Experimental test of airplane boarding methods*  
          [doi:10.1016/j.jairtraman.2011.10.002](https://doi.org/10.1016/j.jairtraman.2011.10.002)  
          Real-world validation with 72 passengers in a mock 757. Steffen method
          was 2× faster than back-to-front and 20–30 % faster than random.
        """)

        st.subheader("Extensions and comparisons")
        st.markdown("""
        - **Milne, R.J. & Kelly, A.R. (2014)**  
          *A new method for boarding passengers onto an airplane*  
          [doi:10.1016/j.jairtraman.2014.08.002](https://doi.org/10.1016/j.jairtraman.2014.08.002)  
          Proposes assigning passengers with more baggage to seats that fit the
          Steffen pattern best — a 2–3 % improvement over pure Steffen.

        - **Milne, R.J. & Salari, M. (2016)**  
          *Optimization of assigning passengers to seats on airplanes based on
          their carry-on luggage*  
          [doi:10.1016/j.jairtraman.2016.02.007](https://doi.org/10.1016/j.jairtraman.2016.02.007)  
          Mixed Integer Programming for baggage-aware seat assignment.

        - **Erland, S. et al. (2024)**  
          *Boarding with slow passengers first*  
          Shows that letting high-baggage passengers board first reduces overall
          queue blocking — counterintuitive but mathematically sound.
        """)

        st.subheader("Accessible overviews")
        st.markdown("""
        - **Wikipedia: Airline boarding problem**  
          [en.wikipedia.org/wiki/Airline_boarding_problem](https://en.wikipedia.org/wiki/Airline_boarding_problem)  
          Good summary of all major methods with timing comparisons.

        - **Scientific American — The Science of Boarding**  
          [scientificamerican.com — search "airplane boarding"](https://www.scientificamerican.com/search/?q=airplane+boarding)  

        - **CGP Grey — The Better Boarding Method Airlines Won't Use (YouTube)**  
          [youtube.com/watch?v=oAHbLRjF0vo](https://www.youtube.com/watch?v=oAHbLRjF0vo)  
          Clear visual explanation of why Steffen works and why airlines don't use it.
        """)

        st.subheader("CSV manifest format")
        st.code(
            "passenger_id,name,row,column,group_id,baggage_size\n"
            "P0001,Alice Smith,28,A,G001,large\n"
            "P0002,Bob Smith,28,B,G001,small\n"
            "P0003,Carol Jones,24,F,,none",
            language="csv",
        )
        st.caption(
            "`group_id` may be blank for solo travellers.  "
            "`baggage_size` must be one of: `none`, `small`, `large`."
        )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
if st.session_state.app_mode == "Planner":
    render_planner()
elif st.session_state.app_mode == "Gate Mode":
    render_gate()
elif st.session_state.app_mode == "Simulation":
    render_simulation()
else:
    render_help()
