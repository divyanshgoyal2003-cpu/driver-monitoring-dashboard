"""
Analytics dashboard for recorded monitoring sessions.

    streamlit run dashboard/dashboard.py
    streamlit run dashboard/dashboard.py -- --db path/to/other.db

Sessions are recorded by the live monitor (python -m app.main).
"""

import argparse
import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

# Streamlit runs this file as a script; make the project importable.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.config import DATABASE_PATH, SAFE_SCORE, CAUTION_SCORE  # noqa: E402
from database.database import Database  # noqa: E402

# Chart colours: slots 1-2 of the reference categorical palette (validated
# for colour-vision deficiency in both modes), recessive chrome, and the
# fixed status colours for risk levels (always shown with a text label).
PALETTES = {
    "light": {
        "series": ["#2a78d6", "#eb6834"],
        "ink": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
    },
    "dark": {
        "series": ["#3987e5", "#d95926"],
        "ink": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
    },
}

RISK_LEVELS = ["SAFE", "CAUTION", "DANGER"]
RISK_COLORS = ["#0ca30c", "#fab219", "#d03b3b"]

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DATABASE_PATH)

    # Streamlit passes only the arguments after "--".
    args, _ = parser.parse_known_args()

    return args


def theme_mode():

    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


def format_duration(seconds):

    if seconds is None or pd.isna(seconds):
        return "--"

    seconds = int(round(seconds))

    return f"{seconds // 60}:{seconds % 60:02d}"


def load(path, session_id=None):

    database = Database(path)

    try:
        sessions = database.sessions()

        if session_id is None:
            return sessions, None, None

        return (
            sessions,
            database.metrics(session_id),
            database.events(session_id),
        )

    finally:
        database.close()


def session_label(row):

    source = row["source"]

    if source and not str(source).isdigit():
        source = os.path.basename(source)
    else:
        source = f"camera {source}"

    label = (
        f"#{row['id']}  ·  {row['started_at'].replace('T', ' ')}  ·  "
        f"{format_duration(row['duration'])}  ·  {source}"
    )

    if pd.isna(row["ended_at"]):
        label += "  (in progress)"

    return label


class Charts:
    """Altair charts styled for the current light/dark mode."""

    def __init__(self, mode, duration):

        self.palette = PALETTES[mode]

        # Long sessions read better in minutes.
        if duration and duration > 180:
            self.time_field, self.time_title = "minutes", "Time (min)"
        else:
            self.time_field, self.time_title = "seconds", "Time (s)"

    def style(self, chart, height=220):

        p = self.palette

        return (
            chart.properties(height=height)
            .configure(background="transparent", font=FONT)
            .configure_view(strokeWidth=0)
            .configure_axis(
                gridColor=p["grid"],
                domainColor=p["baseline"],
                tickColor=p["baseline"],
                labelColor=p["muted"],
                titleColor=p["ink"],
                titleFontWeight="normal",
            )
            .configure_legend(
                labelColor=p["ink"],
                titleColor=p["ink"],
                orient="top",
            )
        )

    def line(self, data, field, title, domain=None, thresholds=(),
             overlay=None, overlay_title=None):
        """
        One measure over time, with a hover crosshair and tooltip.
        thresholds: (value, label) pairs drawn as labelled hairlines.
        overlay: another column drawn as a dashed second line, for a
        reference that moves (the learned EAR threshold, say).
        """

        p = self.palette
        color = p["series"][0]

        hover = alt.selection_point(
            nearest=True, on="pointerover", fields=[self.time_field],
            empty=False
        )

        x = alt.X(f"{self.time_field}:Q", title=self.time_title)
        y = alt.Y(
            f"{field}:Q",
            title=title,
            scale=alt.Scale(domain=domain) if domain else alt.Undefined,
        )

        base = alt.Chart(data).encode(x=x)

        line = base.mark_line(strokeWidth=2, color=color).encode(y=y)

        # Invisible wide targets so hovering anywhere near works.
        targets = base.mark_point(opacity=0, size=600).encode(
            y=y
        ).add_params(hover)

        point = base.mark_point(
            size=70, filled=True, color=color
        ).encode(
            y=y,
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        )

        rule = base.mark_rule(color=p["muted"]).encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip(f"{self.time_field}:Q", title=self.time_title,
                            format=".1f"),
                alt.Tooltip(f"{field}:Q", title=title, format=".2f"),
            ],
        )

        layers = [line, targets, point, rule]

        if overlay and overlay in data and data[overlay].notna().any():
            layers.append(
                base.mark_line(
                    strokeWidth=1.5, color=p["series"][1], strokeDash=[4, 3]
                ).encode(y=alt.Y(f"{overlay}:Q", title=title))
            )

            last = data.dropna(subset=[overlay]).tail(1)

            layers.append(
                alt.Chart(last).mark_text(
                    align="right", dx=-4, dy=-6, color=p["series"][1],
                    fontSize=11
                ).encode(
                    x=alt.X(f"{self.time_field}:Q"),
                    y=alt.Y(f"{overlay}:Q"),
                    text=alt.value(overlay_title or overlay),
                )
            )

        if thresholds:
            limits = pd.DataFrame(thresholds, columns=["value", "label"])
            limit_base = alt.Chart(limits).encode(y="value:Q")

            layers.append(
                limit_base.mark_rule(color=p["baseline"], strokeWidth=1)
            )
            layers.append(
                limit_base.mark_text(
                    align="left", dx=4, dy=-6, color=p["muted"], fontSize=11
                ).encode(x=alt.value(0), text="label:N")
            )

        return self.style(alt.layer(*layers))

    def angles(self, data):
        """Yaw and pitch on one axis (same unit), with direct labels."""

        p = self.palette

        long = data.melt(
            id_vars=[self.time_field],
            value_vars=["yaw", "pitch"],
            var_name="angle",
            value_name="degrees",
        )
        long["angle"] = long["angle"].str.capitalize()

        color = alt.Color(
            "angle:N",
            title=None,
            scale=alt.Scale(domain=["Yaw", "Pitch"], range=p["series"]),
        )

        x = alt.X(f"{self.time_field}:Q", title=self.time_title)
        y = alt.Y("degrees:Q", title="Degrees")

        lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
            x=x, y=y, color=color,
            tooltip=[
                alt.Tooltip("angle:N", title="Angle"),
                alt.Tooltip(f"{self.time_field}:Q", title=self.time_title,
                            format=".1f"),
                alt.Tooltip("degrees:Q", title="Degrees", format=".1f"),
            ],
        )

        last = long.dropna().groupby("angle").tail(1)

        labels = alt.Chart(last).mark_text(
            align="left", dx=6, color=p["ink"], fontSize=11
        ).encode(x=x, y=y, text="angle:N")

        zero = alt.Chart(pd.DataFrame({"degrees": [0]})).mark_rule(
            color=p["baseline"]
        ).encode(y="degrees:Q")

        return self.style(alt.layer(zero, lines, labels))

    def series(self, data, fields, labels, title, domain=None):
        """Several measures sharing one unit, with direct labels."""

        p = self.palette

        long = data.melt(
            id_vars=[self.time_field],
            value_vars=fields,
            var_name="series",
            value_name="value",
        ).dropna(subset=["value"])

        long["series"] = long["series"].map(dict(zip(fields, labels)))

        x = alt.X(f"{self.time_field}:Q", title=self.time_title)
        y = alt.Y(
            "value:Q",
            title=title,
            scale=alt.Scale(domain=domain) if domain else alt.Undefined,
        )
        color = alt.Color(
            "series:N",
            title=None,
            scale=alt.Scale(domain=labels, range=p["series"]),
            legend=None,
        )

        lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
            x=x, y=y, color=color,
            tooltip=[
                alt.Tooltip("series:N", title=""),
                alt.Tooltip(f"{self.time_field}:Q", title=self.time_title,
                            format=".1f"),
                alt.Tooltip("value:Q", title=title, format=".2f"),
            ],
        )

        last = long.groupby("series").tail(1)

        text = alt.Chart(last).mark_text(
            align="left", dx=6, color=p["ink"], fontSize=11
        ).encode(x=x, y=y, text="series:N")

        return self.style(alt.layer(lines, text))

    def bars(self, data, category, value, title, label_format,
             colors=None, order=None, domain=None, integer=False):
        """
        Horizontal bars with value labels at the bar ends. domain fixes
        the value scale (e.g. 0-100 for percentages); integer keeps the
        axis on whole numbers (counts).
        """

        p = self.palette

        if colors:
            color = alt.Color(
                f"{category}:N",
                scale=alt.Scale(domain=order, range=colors),
                legend=None,
            )
        else:
            color = alt.value(p["series"][0])

        y = alt.Y(
            f"{category}:N",
            title=None,
            sort=order if order else "-x",
        )
        x = alt.X(
            f"{value}:Q",
            title=title,
            scale=alt.Scale(domain=domain) if domain else alt.Undefined,
            axis=alt.Axis(tickMinStep=1, format="d") if integer
            else alt.Undefined,
        )

        bars = alt.Chart(data).mark_bar(
            cornerRadiusEnd=4, size=18
        ).encode(
            y=y, x=x, color=color,
            tooltip=[
                alt.Tooltip(f"{category}:N", title=category.capitalize()),
                alt.Tooltip(f"{value}:Q", title=title, format=label_format),
            ],
        )

        labels = alt.Chart(data).mark_text(
            align="left", dx=4, color=p["ink"], fontSize=11
        ).encode(
            y=y, x=x,
            text=alt.Text(f"{value}:Q", format=label_format),
        )

        height = max(60, 30 * len(data))

        return self.style(alt.layer(bars, labels), height=height)


def show_kpis(session, metrics):

    row = st.columns(4)

    row[0].metric("Duration", format_duration(session["duration"]))

    if session["average_score"] is not None and not pd.isna(
            session["average_score"]):
        row[1].metric("Average safety index",
                      f"{session['average_score']:.0f}")
        row[2].metric("Lowest safety index",
                      f"{session['minimum_score']:.0f}")
    else:
        row[1].metric("Average safety index", "--")
        row[2].metric("Lowest safety index", "--")

    if len(metrics):
        danger = (metrics["risk_level"] == "DANGER").mean() * 100
        row[3].metric("Time in danger", f"{danger:.0f}%")
    else:
        row[3].metric("Time in danger", "--")

    row = st.columns(5)

    minutes = (session["duration"] or 0) / 60
    blinks = int(session["blinks"] or 0)

    row[0].metric(
        "Blinks", blinks,
        help=f"{blinks / minutes:.0f} per minute" if minutes > 0 else None,
    )
    row[1].metric("Yawns", int(session["yawns"] or 0))
    row[2].metric("Microsleeps", int(session["microsleeps"] or 0))
    row[3].metric("Distractions", int(session["distractions"] or 0))
    row[4].metric("Phone uses", int(session["phone_uses"] or 0))


def share_by(metrics, column, order=None):
    """Share of logged time per value of `column`, in percent."""

    counts = metrics[column].dropna().value_counts(normalize=True) * 100

    if order:
        counts = counts.reindex(order, fill_value=0.0)

    return counts.rename_axis(column).reset_index(name="percent")


def has_data(metrics, column):
    return column in metrics and metrics[column].notna().any()


def scoring_mode(metrics):
    """
    "dynamic", "static", or None for a session recorded before the
    weighting layers existed.
    """

    if not has_data(metrics, "weights_mode"):
        return None

    return metrics["weights_mode"].dropna().iloc[-1]


def show_weighting(metrics, charts):
    """
    What the risk weights were doing: how much the situation amplified
    each penalty, and how much light the camera had to work with.
    Nothing is shown for a session scored with the flat table.
    """

    mode = scoring_mode(metrics)

    if mode is None:
        return

    st.subheader("Risk weighting")

    if mode != "dynamic":
        st.caption(
            "This session was scored with the flat penalty table "
            "(`--static-weights`), so no weighting was applied."
        )
        return

    left, right = st.columns(2)

    with left:
        if has_data(metrics, "context_fatigue"):
            st.altair_chart(
                charts.series(
                    metrics,
                    ["context_fatigue", "context_attention"],
                    ["Fatigue", "Attention"],
                    "Context multiplier",
                ),
                theme=None,
                width="stretch",
            )
            st.caption(
                "How much the time of day, time at the wheel and cabin "
                "light raised each penalty. 1.0 is the flat baseline."
            )
        else:
            st.caption("No context recorded.")

    with right:
        if has_data(metrics, "brightness"):
            st.altair_chart(
                charts.line(metrics, "brightness", "Cabin brightness",
                            domain=[0, 255]),
                theme=None,
                width="stretch",
            )
            st.caption(
                "Mean frame luminance. A dark cabin weighs attention "
                "more and the eye measurements less."
            )
        else:
            st.caption("No brightness recorded.")


def main():

    st.set_page_config(
        page_title="Driver Monitoring Analytics",
        page_icon=":material/monitoring:",
        layout="wide",
    )

    args = parse_args()

    st.title("Driver Monitoring Analytics")

    if not os.path.exists(args.db):
        st.info(
            f"No sessions recorded yet ({args.db} does not exist). "
            "Run the live monitor with `python -m app.main` to record "
            "one, then refresh this page."
        )
        return

    sessions, _, _ = load(args.db)

    if sessions.empty:
        st.info("The database has no sessions yet. Run "
                "`python -m app.main` to record one.")
        return

    labels = {row["id"]: session_label(row) for _, row in sessions.iterrows()}

    session_id = st.selectbox(
        "Session",
        options=list(labels),
        format_func=labels.get,
    )

    sessions, metrics, events = load(args.db, session_id)
    session = sessions[sessions["id"] == session_id].iloc[0]

    if metrics.empty:
        st.warning("This session has no recorded measurements.")
        return

    metrics["seconds"] = metrics["elapsed"]
    metrics["minutes"] = metrics["elapsed"] / 60

    charts = Charts(theme_mode(), session["duration"])

    # Say which scoring engine produced these numbers, so two runs of
    # the same drive cannot be mistaken for each other.
    mode = scoring_mode(metrics)

    if mode is not None:
        st.caption(
            f"Scored with **{mode}** weighting"
            + ("" if mode == "dynamic" else " (`--static-weights`)")
        )

    show_kpis(session, metrics)

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Safety index")
        st.altair_chart(
            charts.line(
                metrics, "safety_score", "Safety index",
                domain=[0, 100],
                thresholds=[(SAFE_SCORE, f"Safe ≥ {SAFE_SCORE}"),
                            (CAUTION_SCORE, f"Caution ≥ {CAUTION_SCORE}")],
            ),
            theme=None,
            width="stretch",
        )

    with right:
        st.subheader("Time at each risk level")
        st.altair_chart(
            charts.bars(
                share_by(metrics, "risk_level", RISK_LEVELS),
                "risk_level", "percent", "Share of session (%)", ".0f",
                colors=RISK_COLORS, order=RISK_LEVELS, domain=[0, 100],
            ),
            theme=None,
            width="stretch",
        )

    st.subheader("Eyes and mouth")

    columns = st.columns(3)

    # The EAR chart carries the driver's own blink threshold, so the
    # measurement can be read against what it was actually compared to.
    small_multiples = [
        ("ear", "Eye aspect ratio", None, "ear_threshold", "threshold"),
        ("perclos", "PERCLOS (eyes closed share)", [0, 1], None, None),
        ("mar", "Mouth aspect ratio", None, None, None),
    ]

    for column, (field, title, domain, overlay, overlay_title) in zip(
            columns, small_multiples):
        with column:
            if has_data(metrics, field):
                st.altair_chart(
                    charts.line(metrics, field, title, domain=domain,
                                overlay=overlay,
                                overlay_title=overlay_title),
                    theme=None,
                    width="stretch",
                )
            else:
                st.caption(f"{title}: no data (face not visible)")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Head pose")

        if has_data(metrics, "yaw"):
            st.altair_chart(charts.angles(metrics), theme=None,
                            width="stretch")
        else:
            st.caption("No head pose data (face not visible).")

    with right:
        st.subheader("Emotion")

        if has_data(metrics, "emotion"):
            st.altair_chart(
                charts.bars(share_by(metrics, "emotion"), "emotion",
                            "percent", "Share of session (%)", ".0f",
                            domain=[0, 100]),
                theme=None,
                width="stretch",
            )
        else:
            st.caption("No emotion data (face not visible).")

    show_weighting(metrics, charts)

    st.subheader("Events")

    if events.empty:
        st.caption("No events in this session.")
    else:
        left, right = st.columns([1, 2])

        counts = (
            events["event_type"].value_counts()
            .rename_axis("event")
            .reset_index(name="count")
        )

        with left:
            st.altair_chart(
                charts.bars(counts, "event", "count", "Occurrences", "d",
                            integer=True),
                theme=None,
                width="stretch",
            )

        with right:
            table = events.assign(
                time=events["elapsed"].map(format_duration)
            )[["time", "event_type", "severity", "message"]]

            st.dataframe(
                table.rename(columns={"event_type": "event"}),
                hide_index=True,
                width="stretch",
            )

    with st.expander("All measurements (table view)"):
        st.dataframe(metrics.drop(columns=["seconds", "minutes"]),
                     hide_index=True, width="stretch")
        st.download_button(
            "Download CSV",
            metrics.to_csv(index=False).encode("utf-8"),
            file_name=f"session_{session_id}.csv",
            mime="text/csv",
        )

    with st.expander("All sessions"):
        st.dataframe(sessions, hide_index=True, width="stretch")


# Streamlit runs the script as __main__; importing it (for tests) does
# not start the app.
if __name__ == "__main__":
    main()
