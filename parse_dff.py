#!/usr/bin/env python3
"""
Unified widefield calcium-imaging (dFF) parser — works for any mouse subject
that has trial-averaged dF/F .mat files (a struct of outcome-category fields,
each an H x W x T array). Handles both layouts seen in this dataset:
  - Carmel: <subject>/<DAY>/<DAY>_dFF_BS_N3.mat            (no phase)
  - Duluth: <subject>/dFF/<Phase>/<DAY>_dFF_BS_N3_TD.mat   (Phase = Naive/Expert)

Usage: python3 parse_dff.py <subject> <src_root> <out_root>

Writes data/<subject>/processed/dFF/dFF_mean_traces.csv (spatially-averaged
ΔF/F time course per day/category/phase) plus one <day>_<category>_peakframe.csv
per category (the H x W spatial map at that category's peak-activity frame).
"""
import sys
import os
import re
import glob
import numpy as np
import pandas as pd
import scipy.io as sio


def parse_dff(subject, src_root, out_root):
    src = os.path.join(src_root, subject)
    out = os.path.join(out_root, subject, "processed", "dFF")
    os.makedirs(out, exist_ok=True)

    dff_files = glob.glob(os.path.join(src, "**", "*_dFF_*.mat"), recursive=True)
    if not dff_files:
        print(f"[{subject}] no dFF files found — skipping imaging.")
        return None

    print(f"[{subject}] found {len(dff_files)} dFF files")
    trace_rows = []
    for f in sorted(dff_files):
        m = re.search(r"(\d{6})", os.path.basename(f))
        if not m:
            continue
        day = m.group(1)
        parent = os.path.basename(os.path.dirname(f))
        phase = parent if parent in ("Naive", "Expert") else ""

        try:
            d = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["dFF"]
        except Exception as e:
            print(f"  [{subject}/{day}] failed to load {f}: {e}")
            continue

        for cat in d._fieldnames:
            arr = np.array(getattr(d, cat), dtype=float)
            if arr.ndim != 3:
                continue
            H, W, T = arr.shape
            flat = arr.reshape(-1, T)
            trace = np.nanmean(flat, axis=0)
            for fr, val in enumerate(trace):
                trace_rows.append({"session": day, "phase": phase, "category": cat,
                                    "frame": fr, "mean_dFF": float(val),
                                    "H": H, "W": W, "n_frames": T})
            peak_frame = int(np.nanargmax(np.nanmean(np.abs(flat), axis=0)))
            tag = f"{day}_{phase}_{cat}" if phase else f"{day}_{cat}"
            np.savetxt(os.path.join(out, f"{tag}_peakframe.csv"), arr[:, :, peak_frame], delimiter=",")
        print(f"  [{subject}/{day}] {parent or '(root)'}: categories = {d._fieldnames}")

    trace_df = pd.DataFrame(trace_rows)
    trace_df.to_csv(os.path.join(out, "dFF_mean_traces.csv"), index=False)
    print(f"[{subject}] dFF mean-trace rows: {len(trace_df)}")
    return trace_df


if __name__ == "__main__":
    subject, src_root, out_root = sys.argv[1], sys.argv[2], sys.argv[3]
    parse_dff(subject, src_root, out_root)
