#!/usr/bin/env python3
"""
Unified behavioral-data parser — works for any mouse subject with the standard
event_history / lick_history / camera_history / parameters / time_stamp / RT
.mat file layout (Austin, Carmel, Eenie, Duluth all share this schema).

Usage:  python3 parse_mouse.py <subject> <src_root> <out_root>
  <src_root> must contain <subject>/YYMMDD/YYMMDD_<type>.mat folders
  <out_root> is where data/<subject>/processed/ tables are written

Scoring method (applied uniformly to every subject, matching the Carmel-Analysis
repo's approach): a trial is scored by whether >=1 lick occurs in the response
window [onset + 1.0s, onset + 2.5s] (1s pre-stimulus baseline + 1s lick window +
0.5s reaction buffer).
  Target + lick -> Hit         Target + no lick -> Miss
  Distractor + lick -> FA      Distractor + no lick -> CR
  Catch(8) + lick -> CatchLick Catch(8) + no lick -> CatchNoLick
d' = z(HitRate) - z(FArate), rates clipped to avoid +/-inf.
"""
import sys
import os
import re
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import norm

STIM_MAP = {1: "Target", 3: "Distractor", 8: "Catch"}
PRE_STIM = 1.0
LICK_WINDOW = 1.0
RT_BUFFER = 0.5
WIN_START = PRE_STIM
WIN_END = PRE_STIM + LICK_WINDOW + RT_BUFFER


def parse_subject(subject, src_root, out_root):
    src = os.path.join(src_root, subject)
    out = os.path.join(out_root, subject, "processed")
    os.makedirs(out, exist_ok=True)

    days = sorted([d for d in os.listdir(src) if re.fullmatch(r"\d{6}(-\d+)?", d)])
    print(f"[{subject}] {len(days)} sessions found: {days}")

    trial_rows, session_rows, param_rows, rt_rows = [], [], [], []
    licks_rows, cameras_rows = [], []

    for day in days:
        d = os.path.join(src, day)
        daytag = day  # keep the raw folder name (handles "230715-2" duplicate sessions)

        # ---------------- event_history ----------------
        eh_path = os.path.join(d, f"{day.split('-')[0]}_event_history.mat")
        if not os.path.exists(eh_path):
            print(f"  [{subject}/{daytag}] no event_history, skipping session")
            continue
        eh = sio.loadmat(eh_path)["event_history"]
        n_trials = eh.shape[1]
        stim_code, outcome_code_raw, stim_dur, iti_s, onset_s = eh[0], eh[1], eh[2], eh[3], eh[4]

        # ---------------- lick_history (collapse to discrete events) ----------------
        lh_path = os.path.join(d, f"{day.split('-')[0]}_lick_history.mat")
        licks = np.array([])
        n_lick_samples = 0
        if os.path.exists(lh_path):
            raw = sio.loadmat(lh_path)["lick_history"].ravel()
            n_lick_samples = raw.size
            if raw.size:
                gaps = np.diff(raw)
                onset_idx = np.r_[0, np.where(gaps > 0.05)[0] + 1]
                licks = raw[onset_idx]
                for t in licks:
                    licks_rows.append({"session": daytag, "lick_time_s": float(t)})

        # ---------------- camera_history ----------------
        ch_path = os.path.join(d, f"{day.split('-')[0]}_camera_history.mat")
        n_frames, cam_fps = 0, np.nan
        if os.path.exists(ch_path):
            cam = sio.loadmat(ch_path)["camera_history"].ravel()
            n_frames = cam.size
            if cam.size > 1:
                cam_fps = 1 / np.median(np.diff(cam))
            for t in cam:
                cameras_rows.append({"session": daytag, "frame_time_s": float(t)})

        # ---------------- parameters ----------------
        p_path = os.path.join(d, f"{day.split('-')[0]}_parameters.mat")
        prow = {"session": daytag}
        if os.path.exists(p_path):
            p = sio.loadmat(p_path)["parameters"][0, 0]
            for name in p.dtype.names:
                v = p[name]
                if v.dtype.kind in "US":
                    prow[name] = str(v[0]) if v.size else None
                else:
                    arr = np.array(v).flatten()
                    if arr.size == 1:
                        prow[name] = arr[0]
                    elif arr.size > 1:
                        prow[name + "_min"] = arr.min()
                        prow[name + "_max"] = arr.max()
                        prow[name + "_n"] = arr.size
        param_rows.append(prow)

        # ---------------- RT.mat (optional, supplementary) ----------------
        rt_path = os.path.join(d, f"{day.split('-')[0]}_RT.mat")
        if os.path.exists(rt_path):
            try:
                rtm = sio.loadmat(rt_path)["RT"][0, 0]
                for cat in ["Target", "Distractor"]:
                    if f"{cat}_RT" in rtm.dtype.names:
                        rt_arr = rtm[f"{cat}_RT"][0]
                        iti_arr = rtm[f"{cat}_ITI"][0] if f"{cat}_ITI" in rtm.dtype.names else []
                        for j in range(len(rt_arr)):
                            rt_rows.append({"session": daytag, "category": cat,
                                             "rt": float(rt_arr[j]),
                                             "iti": float(iti_arr[j]) if j < len(iti_arr) else np.nan})
            except Exception as e:
                print(f"  [{subject}/{daytag}] RT.mat parse skipped: {e}")

        # ---------------- score trials via lick cross-referencing ----------------
        for i in range(n_trials):
            onset = onset_s[i]
            w0, w1 = onset + WIN_START, onset + WIN_END
            licked = bool(np.any((licks >= w0) & (licks <= w1))) if licks.size else False
            stim = STIM_MAP.get(int(round(stim_code[i])), f"Unknown({stim_code[i]})")
            if stim == "Target":
                outcome = "Hit" if licked else "Miss"
            elif stim == "Distractor":
                outcome = "FA" if licked else "CR"
            else:
                outcome = "CatchLick" if licked else "CatchNoLick"
            win_licks = licks[(licks >= w0) & (licks <= w1)] if licks.size else np.array([])
            lat = float(win_licks.min() - (onset + PRE_STIM)) if win_licks.size else np.nan
            trial_rows.append({
                "session": daytag, "trial": i + 1, "stimulus": stim, "outcome": outcome,
                "licked": licked, "lick_latency_s": lat, "trial_onset_s": float(onset),
                "iti_s": float(iti_s[i]), "stim_duration_s": float(stim_dur[i]),
                "outcome_code_raw": float(outcome_code_raw[i]),
            })

        # ---------------- session-level metrics ----------------
        td = pd.DataFrame([t for t in trial_rows if t["session"] == daytag])
        n_t = (td.stimulus == "Target").sum()
        n_d = (td.stimulus == "Distractor").sum()
        n_c = (td.stimulus == "Catch").sum()
        hits = (td.outcome == "Hit").sum()
        miss = (td.outcome == "Miss").sum()
        fa = (td.outcome == "FA").sum()
        cr = (td.outcome == "CR").sum()
        hr = hits / n_t if n_t else np.nan
        far = fa / n_d if n_d else np.nan
        hr_c = min(max(hr, 0.5 / max(n_t, 1)), 1 - 0.5 / max(n_t, 1)) if n_t else np.nan
        far_c = min(max(far, 0.5 / max(n_d, 1)), 1 - 0.5 / max(n_d, 1)) if n_d else np.nan
        dprime = norm.ppf(hr_c) - norm.ppf(far_c) if n_t and n_d else np.nan
        crit = -0.5 * (norm.ppf(hr_c) + norm.ppf(far_c)) if n_t and n_d else np.nan
        session_duration_min = (onset_s.max() - onset_s.min()) / 60 if n_trials else np.nan
        n_licks = int(licks.size)
        lick_rate_hz = n_licks / (session_duration_min * 60) if session_duration_min and n_licks else np.nan

        session_rows.append({
            "session": daytag, "n_trials": int(n_trials),
            "n_target": int(n_t), "n_distractor": int(n_d), "n_catch": int(n_c),
            "hits": int(hits), "misses": int(miss), "false_alarms": int(fa), "correct_rejections": int(cr),
            "hit_rate": hr, "fa_rate": far, "accuracy": (hits + cr) / len(td) if len(td) else np.nan,
            "dprime": dprime, "criterion": crit,
            "mean_lick_latency_s": td.lick_latency_s.mean(),
            "session_duration_min": session_duration_min,
            "n_licks": n_licks, "lick_rate_hz": lick_rate_hz,
            "n_camera_frames": int(n_frames), "camera_fps_est": cam_fps,
        })

    # ---------------- write outputs ----------------
    trials_df = pd.DataFrame(trial_rows)
    licks_df = pd.DataFrame(licks_rows)
    cameras_df = pd.DataFrame(cameras_rows)
    params_df = pd.DataFrame(param_rows)
    rt_df = pd.DataFrame(rt_rows)
    summary_df = pd.DataFrame(session_rows)
    # sort sessions chronologically (strip any "-2" suffix for the date, keep the tag for display)
    summary_df["_date_sort"] = summary_df["session"].str.slice(0, 6)
    summary_df = summary_df.sort_values(["_date_sort", "session"]).drop(columns="_date_sort").reset_index(drop=True)
    summary_df["day_number"] = range(1, len(summary_df) + 1)
    trials_df = trials_df.merge(summary_df[["session", "day_number"]], on="session", how="left")

    trials_df.to_parquet(os.path.join(out, "trials.parquet"), index=False)
    licks_df.to_parquet(os.path.join(out, "licks.parquet"), index=False)
    cameras_df.to_parquet(os.path.join(out, "cameras.parquet"), index=False)
    params_df.to_csv(os.path.join(out, "parameters.csv"), index=False)
    rt_df.to_parquet(os.path.join(out, "rt_struct.parquet"), index=False)
    summary_df.to_csv(os.path.join(out, "session_summary.csv"), index=False)

    print(f"[{subject}] wrote: trials={trials_df.shape} licks={licks_df.shape} "
          f"cameras={cameras_df.shape} params={params_df.shape} rt={rt_df.shape} "
          f"summary={summary_df.shape}")
    return summary_df


if __name__ == "__main__":
    subject, src_root, out_root = sys.argv[1], sys.argv[2], sys.argv[3]
    df = parse_subject(subject, src_root, out_root)
    print(df[["session", "n_trials", "hit_rate", "fa_rate", "accuracy", "dprime"]].to_string(index=False))
