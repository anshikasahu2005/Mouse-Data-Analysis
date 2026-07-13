"""
Multi-Mouse Behavioral Dashboard
----------------------------------
Auto-discovers subjects from ./data/<subject>/processed/ — every subject uses
the same standardized schema (produced by parse_mouse.py / parse_dff.py), so
adding a new mouse is just: run the parser, drop the output folder in data/,
no code changes needed. A subject's "Calcium Imaging" tab appears automatically
if it has a processed/dFF/ folder.

Scoring: every subject's trials are scored Hit/Miss/FA/CR/CatchLick/CatchNoLick
by cross-referencing lick timing against each trial's response window
(onset + 1.0-2.5s) — the same method for all mice, so metrics are comparable
across subjects.
"""

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Multi-Mouse Behavioral Dashboard", layout="wide",
                    initial_sidebar_state="expanded")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "data")

OUTCOME_COLORS = {"Hit": "#2ca02c", "Miss": "#d62728", "FA": "#ff7f0e", "CR": "#1f77b4",
                   "CatchLick": "#9467bd", "CatchNoLick": "#8c564b"}
STIM_COLORS = {"Target": "#4C72B0", "Distractor": "#DD8452", "Catch": "#8C8C8C"}


def discover_subjects():
    if not os.path.isdir(DATA_ROOT):
        return []
    subs = []
    for name in sorted(os.listdir(DATA_ROOT)):
        p = os.path.join(DATA_ROOT, name, "processed")
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "session_summary.csv")):
            subs.append(name)
    return subs


def has_imaging(subject):
    p = os.path.join(DATA_ROOT, subject, "processed", "dFF", "dFF_mean_traces.csv")
    return os.path.exists(p) and os.path.getsize(p) > 0


@st.cache_data
def load_subject(subject):
    p = os.path.join(DATA_ROOT, subject, "processed")
    trials = pd.read_parquet(os.path.join(p, "trials.parquet"))
    licks = pd.read_parquet(os.path.join(p, "licks.parquet"))
    cameras = pd.read_parquet(os.path.join(p, "cameras.parquet"))
    params = pd.read_csv(os.path.join(p, "parameters.csv"))
    rt_struct = pd.read_parquet(os.path.join(p, "rt_struct.parquet"))
    summary = pd.read_csv(os.path.join(p, "session_summary.csv"))
    for df in (trials, licks, cameras, params, summary):
        df["session"] = df["session"].astype(str)
    summary = summary.sort_values("day_number").reset_index(drop=True)
    return trials, licks, cameras, params, rt_struct, summary


@st.cache_data
def load_dff_traces(subject):
    p = os.path.join(DATA_ROOT, subject, "processed", "dFF", "dFF_mean_traces.csv")
    df = pd.read_csv(p)
    df["session"] = df["session"].astype(str)
    df["phase"] = df["phase"].fillna("").astype(str)
    return df


@st.cache_data
def load_dff_peakframe(subject, session, phase, category):
    tag = f"{session}_{phase}_{category}" if phase else f"{session}_{category}"
    p = os.path.join(DATA_ROOT, subject, "processed", "dFF", f"{tag}_peakframe.csv")
    return np.loadtxt(p, delimiter=",") if os.path.exists(p) else None


# ============================================================================
subjects = discover_subjects()
st.sidebar.title("🐭 Multi-Mouse Behavioral Dashboard")

if not subjects:
    st.error("No subjects found under ./data/. Run parse_mouse.py to add one.")
    st.stop()

subject = st.sidebar.selectbox("Subject", subjects)
trials, licks, cameras, params, rt_struct, summary = load_subject(subject)
imaging = has_imaging(subject)

st.sidebar.caption(f"{summary.shape[0]} sessions · {int(summary.n_trials.sum()):,} trials"
                    f"{' · has calcium imaging' if imaging else ''}")

sess_labels = summary["session"].tolist()
selected_sessions = st.sidebar.multiselect("Sessions", sess_labels, default=sess_labels)
selected_sessions = selected_sessions or sess_labels
stim_filter = st.sidebar.multiselect("Stimulus type", ["Target", "Distractor", "Catch"],
                                      default=["Target", "Distractor", "Catch"])
st.sidebar.markdown("---")
st.sidebar.caption(
    "Raw widefield imaging (`.tif`) and behavior video (`.avi`) files are not included — "
    "too large for interactive analysis. Outcomes are scored by cross-referencing lick "
    "timing against each trial's response window (onset + 1.0-2.5s), consistently across "
    "every subject."
)

t_f = trials[trials.session.isin(selected_sessions) & trials.stimulus.isin(stim_filter)]
s_f = summary[summary.session.isin(selected_sessions)]
l_f = licks[licks.session.isin(selected_sessions)]

tab_names = ["📊 Overview", "🎯 Trial Outcomes", "⏱️ Lick Latency", "👅 Licking",
             "📷 Session Timing", "⚙️ Parameters", "🔎 Raw Data"]
if imaging:
    tab_names.append("🧠 Calcium Imaging (ΔF/F)")
tabs = st.tabs(tab_names)

# ============================================================= OVERVIEW ===
with tabs[0]:
    st.header(f"Overview — {subject}")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sessions", f"{s_f.shape[0]}")
    k2.metric("Total trials", f"{int(s_f.n_trials.sum()):,}")
    k3.metric("Mean accuracy", f"{s_f.accuracy.mean()*100:.0f}%" if s_f.accuracy.notna().any() else "—")
    k4.metric("Best d′", f"{s_f.dprime.max():.2f}" if s_f.dprime.notna().any() else "—")
    k5.metric("Total licks", f"{int(s_f.n_licks.sum()):,}")

    st.subheader("Learning curve")
    m = st.multiselect("Metrics", ["hit_rate", "fa_rate", "accuracy", "dprime"],
                        default=["hit_rate", "fa_rate", "dprime"], key="metrics_ms")
    if m:
        long = s_f.melt(id_vars=["day_number", "session"], value_vars=m, var_name="metric", value_name="value")
        fig = px.line(long, x="day_number", y="value", color="metric", markers=True,
                      hover_data=["session"], labels={"day_number": "Training day #"})
        fig.add_hline(y=0.5, line_dash="dot", line_color="grey")
        st.plotly_chart(fig, use_container_width=True, key="overview_learning")
    st.caption("Hit/Miss/FA/CR scored from lick timing in each trial's response window; "
               "d′ and accuracy follow standard signal-detection formulas (rates clipped to avoid ±∞).")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Trial composition per session")
        stacked = s_f.melt(id_vars=["day_number", "session"],
                            value_vars=["hits", "misses", "false_alarms", "correct_rejections"],
                            var_name="outcome", value_name="count")
        fig = px.bar(stacked, x="day_number", y="count", color="outcome",
                     labels={"day_number": "Training day #"}, hover_data=["session"])
        st.plotly_chart(fig, use_container_width=True, key="overview_composition")
    with c2:
        st.subheader("Session duration & lick rate")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=s_f.day_number, y=s_f.session_duration_min, name="Duration (min)",
                              marker_color="#4C72B0"))
        fig.add_trace(go.Scatter(x=s_f.day_number, y=s_f.lick_rate_hz, name="Lick rate (Hz)",
                                  mode="lines+markers", marker_color="#C44E52", yaxis="y2"))
        fig.update_layout(xaxis_title="Training day #", yaxis=dict(title="Duration (min)"),
                           yaxis2=dict(title="Lick rate (Hz)", overlaying="y", side="right"),
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True, key="overview_duration")

# ======================================================= TRIAL OUTCOMES ===
with tabs[1]:
    st.header("Trial Outcomes")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Outcome mix per session")
        outc = t_f.groupby(["day_number", "session", "outcome"]).size().reset_index(name="count")
        fig = px.bar(outc, x="day_number", y="count", color="outcome", color_discrete_map=OUTCOME_COLORS,
                     labels={"day_number": "Training day #"})
        st.plotly_chart(fig, use_container_width=True, key="outcomes_mix")
    with c2:
        st.subheader("Overall breakdown")
        pie = t_f.outcome.value_counts().reset_index()
        pie.columns = ["outcome", "count"]
        fig = px.pie(pie, names="outcome", values="count", color="outcome",
                     color_discrete_map=OUTCOME_COLORS, hole=0.4)
        st.plotly_chart(fig, use_container_width=True, key="outcomes_pie")

    st.subheader("Response criterion (bias) per session")
    fig = px.line(s_f, x="day_number", y="criterion", markers=True, labels={"day_number": "Training day #"})
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    st.plotly_chart(fig, use_container_width=True, key="outcomes_criterion")

    st.subheader("Session performance table")
    show_cols = ["session", "n_trials", "n_target", "n_distractor", "n_catch", "hits", "misses",
                 "false_alarms", "correct_rejections", "hit_rate", "fa_rate", "accuracy", "dprime", "criterion"]
    pct_cols = ["hit_rate", "fa_rate", "accuracy"]
    table_df = s_f[show_cols].copy()
    table_df[pct_cols] = (table_df[pct_cols] * 100).round(1)
    st.dataframe(table_df, use_container_width=True, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(c, format="%.1f%%") for c in pct_cols})
    st.download_button("⬇ Download session_summary.csv", s_f.to_csv(index=False),
                        f"{subject}_session_summary.csv", key="outcomes_dl")

# =========================================================== LICK LATENCY=
with tabs[2]:
    st.header("Lick Latency")
    st.caption("Latency = time from stimulus onset to first in-window lick. Only defined for trials with a lick.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Latency distribution by outcome")
        lat_df = t_f.dropna(subset=["lick_latency_s"])
        fig = px.histogram(lat_df, x="lick_latency_s", color="outcome", nbins=40, opacity=0.7,
                            color_discrete_map=OUTCOME_COLORS, barmode="overlay",
                            labels={"lick_latency_s": "Latency (s)"})
        st.plotly_chart(fig, use_container_width=True, key="lat_hist")
    with c2:
        st.subheader("Latency by stimulus type")
        fig = px.box(lat_df, x="stimulus", y="lick_latency_s", color="stimulus", color_discrete_map=STIM_COLORS,
                     labels={"lick_latency_s": "Latency (s)"})
        st.plotly_chart(fig, use_container_width=True, key="lat_box")

    st.subheader("Median latency across sessions")
    fig = px.line(s_f, x="day_number", y="mean_lick_latency_s", markers=True,
                  labels={"day_number": "Training day #", "mean_lick_latency_s": "Mean latency (s)"})
    st.plotly_chart(fig, use_container_width=True, key="lat_trend")

    if not rt_struct.empty:
        st.subheader("Supplementary: RT.mat struct (Target/Distractor RT vs. ITI)")
        st.caption(f"Only logged for sessions {', '.join(sorted(rt_struct.session.unique()))} in the raw data.")
        fig = px.scatter(rt_struct, x="iti", y="rt", color="category", opacity=0.6, facet_col="session",
                          labels={"iti": "ITI (s)", "rt": "RT (s)"})
        st.plotly_chart(fig, use_container_width=True, key="lat_rt_struct")

# ================================================================ LICKING=
with tabs[3]:
    st.header("Licking Behavior")
    st.subheader("Lick rate across sessions")
    fig = px.bar(s_f, x="day_number", y="lick_rate_hz", hover_data=["session"],
                 labels={"day_number": "Training day #", "lick_rate_hz": "Lick rate (Hz)"})
    st.plotly_chart(fig, use_container_width=True, key="lick_rate_bar")

    st.subheader("Lick raster (select a session)")
    sess_for_raster = st.selectbox("Session", selected_sessions, key="raster_sess",
                                    index=len(selected_sessions) - 1)
    licks_sess = licks[licks.session == sess_for_raster]
    if licks_sess.empty:
        st.info("No lick_history recorded for this session in the raw data.")
    else:
        fig = px.strip(licks_sess, x="lick_time_s", labels={"lick_time_s": "Session time (s)"})
        fig.update_traces(marker=dict(size=3, opacity=0.5))
        fig.update_layout(yaxis_visible=False, height=250)
        st.plotly_chart(fig, use_container_width=True, key="lick_raster")

        st.subheader(f"Lick rate over time — session {sess_for_raster}")
        bins = np.arange(0, licks_sess.lick_time_s.max() + 10, 10)
        counts, edges = np.histogram(licks_sess.lick_time_s, bins=bins)
        rate_df = pd.DataFrame({"t": edges[:-1], "licks_per_10s": counts})
        fig2 = px.area(rate_df, x="t", y="licks_per_10s", labels={"t": "Session time (s)"})
        st.plotly_chart(fig2, use_container_width=True, key="lick_rate_time")

    missing = [s for s in selected_sessions if s not in licks.session.unique()]
    if missing:
        st.caption(f"Sessions with no lick_history file: {', '.join(sorted(missing))}")

# ========================================================= SESSION TIMING=
with tabs[4]:
    st.header("Session Timing & Camera Sync")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Session duration (min)")
        fig = px.bar(s_f, x="day_number", y="session_duration_min", hover_data=["session"],
                     labels={"day_number": "Training day #", "session_duration_min": "Duration (min)"})
        st.plotly_chart(fig, use_container_width=True, key="timing_duration")
    with c2:
        st.subheader("Estimated camera frame rate")
        fig = px.bar(s_f, x="day_number", y="camera_fps_est", hover_data=["session", "n_camera_frames"],
                     labels={"day_number": "Training day #", "camera_fps_est": "FPS"})
        st.plotly_chart(fig, use_container_width=True, key="timing_fps")

    st.subheader("Camera frame timing (select a session)")
    sess_for_cam = st.selectbox("Session ", selected_sessions, key="cam_sess", index=len(selected_sessions) - 1)
    cam_sess = cameras[cameras.session == sess_for_cam].sort_values("frame_time_s")
    if len(cam_sess) > 1:
        intervals = np.diff(cam_sess.frame_time_s.values)
        fig = px.histogram(x=intervals, nbins=50, labels={"x": "Inter-frame interval (s)"})
        st.plotly_chart(fig, use_container_width=True, key="timing_intervals")
        st.caption(f"{len(cam_sess):,} frames · median interval {np.median(intervals)*1000:.1f} ms "
                   f"({1/np.median(intervals):.2f} fps) · max gap {intervals.max():.2f} s")

# ============================================================= PARAMETERS=
with tabs[5]:
    st.header("Task Parameters Across Sessions")
    p_f = params[params.session.isin(selected_sessions)].merge(
        s_f[["session", "day_number"]], on="session").sort_values("day_number")
    st.dataframe(p_f.drop(columns=["day_number"]), use_container_width=True, hide_index=True)

    numeric_candidates = [c for c in ["No_Lick_Interval_min", "No_Lick_Interval_max", "Total_Rewards",
                                       "Lick_Window", "Solenoid_Open", "Left_Stimulus_Duration",
                                       "Right_Stimulus_Duration"] if c in p_f.columns]
    if numeric_candidates:
        st.subheader("Parameter trends across days")
        chosen = st.multiselect("Parameters to plot", numeric_candidates, default=numeric_candidates[:2],
                                 key="param_ms")
        if chosen:
            melt = p_f.melt(id_vars=["day_number", "session"], value_vars=chosen)
            fig = px.line(melt, x="day_number", y="value", color="variable", markers=True,
                         labels={"day_number": "Training day #", "value": "Value"})
            st.plotly_chart(fig, use_container_width=True, key="param_trend")

# ==================================================================== RAW=
with tabs[6]:
    st.header("Raw Trial-Level Data Explorer")
    st.caption("Filtered by the sidebar session/stimulus selection.")
    st.dataframe(t_f.sort_values(["session", "trial"]), use_container_width=True, hide_index=True, height=500)
    st.download_button("⬇ Download filtered trials as CSV", t_f.to_csv(index=False).encode(),
                        file_name=f"{subject}_trials_filtered.csv", mime="text/csv", key="raw_dl")

# ============================================================= IMAGING ===
if imaging:
    with tabs[7]:
        st.header("Widefield Calcium Imaging — ΔF/F")
        dff = load_dff_traces(subject)
        imdays = sorted(dff.session.unique())
        c1, c2 = st.columns(2)
        day = c1.selectbox("Day", imdays, key="dff_day")
        phases = sorted(dff[dff.session == day].phase.unique())
        phase = c2.selectbox("Phase", phases, key="dff_phase") if len(phases) > 1 or phases != [""] else ""
        d = dff[(dff.session == day) & (dff.phase == phase)]

        st.markdown("**Spatially-averaged ΔF/F time course, by trial outcome** "
                    "(each trace = mean over all cortical pixels).")
        title = f"{day}" + (f" ({phase})" if phase else "") + " — mean cortical ΔF/F per outcome category"
        fig = px.line(d, x="frame", y="mean_dFF", color="category", title=title)
        st.plotly_chart(fig, use_container_width=True, key="dff_line")

        st.subheader("Peak-activity cortical maps")
        cats = sorted(d.category.unique())
        chosen = st.multiselect("Categories", cats,
                                 default=[c for c in ["Hits", "Misses", "FA", "CR"] if c in cats],
                                 key="dff_cats")
        cols = st.columns(min(4, max(1, len(chosen))))
        for i, cat in enumerate(chosen):
            arr = load_dff_peakframe(subject, day, phase, cat)
            if arr is not None:
                fig = px.imshow(arr, color_continuous_scale="RdBu_r", origin="upper", title=cat,
                                zmin=-np.nanmax(np.abs(arr)), zmax=np.nanmax(np.abs(arr)))
                fig.update_layout(coloraxis_showscale=False, height=260, margin=dict(l=0, r=0, t=30, b=0))
                cols[i % len(cols)].plotly_chart(fig, use_container_width=True, key=f"dff_peak_{cat}")
        st.caption("Maps show the ΔF/F image at each category's peak-activity frame (red = activation).")

        if "Naive" in phases and "Expert" in phases:
            st.subheader("Naive vs. Expert — same day comparison")
            st.caption("This subject has imaging split into Naive/Expert training phases.")
