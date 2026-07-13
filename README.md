# Multi-Mouse Behavioral Dashboard

One Streamlit app for any number of head-fixed mouse Go/No-Go widefield-imaging
cohorts. Pick a subject in the sidebar — the app auto-discovers whoever is in
`data/`, so adding a new mouse never requires touching `app.py`.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Current subjects

| Subject | Sessions | Dates | Imaging (ΔF/F)? |
|---|---|---|---|
| Austin | 18 | May 2023 | No |
| Carmel | 15 | Jul 2023 | Yes — 1 day (230712) |
| Eenie | 18 | Jul-Aug 2023 | No |
| Duluth | 15 | Jul 2023 | Yes — 12 days, Naive/Expert split |

Every subject is scored the same way: a trial counts as a **Hit/Miss** (Target
stimulus) or **FA/CR** (Distractor stimulus) by checking whether a lick
landed inside the response window (`onset + 1.0s` to `onset + 2.5s`). d-prime
and criterion follow standard signal-detection formulas. This method is
applied uniformly across all four subjects, so metrics are directly
comparable — earlier versions of this dashboard scored Austin differently;
that's been corrected here.

## Adding mouse #5 (and beyond)

1. Get the subject's raw `.mat` files into a folder shaped like
   `<subject>/<YYMMDD>/<YYMMDD>_event_history.mat` (+ `lick_history`,
   `camera_history`, `parameters`, `time_stamp`, optionally `RT.mat`).
   If they have imaging, `.mat` dFF files can either sit inside each day's
   folder (`<subject>/<YYMMDD>/<YYMMDD>_dFF_*.mat`) or in a separate
   `<subject>/dFF/<Phase>/<YYMMDD>_dFF_*.mat` tree (Phase = e.g. Naive/Expert)
   — both layouts are handled automatically.
2. Run the parsers:
   ```bash
   python3 parse_mouse.py <Subject> <path-to-parent-of-subject-folder> data
   python3 parse_dff.py <Subject> <path-to-parent-of-subject-folder> data   # only if they have imaging
   ```
   This writes `data/<Subject>/processed/...`.
3. `streamlit run app.py` — the new subject shows up in the sidebar
   automatically, with the Calcium Imaging tab appearing only if `parse_dff.py`
   found anything.

No other files need to change. `app.py` reads whatever tables exist per
subject and adapts the tab set accordingly.

## Folder layout

```
app.py                    dashboard — auto-discovers subjects from data/
parse_mouse.py            raw .mat -> standardized behavioral tables (any subject)
parse_dff.py              raw .mat -> standardized imaging tables (any subject with dFF)
requirements.txt
data/
  Austin/processed/       trials.parquet, session_summary.csv, licks.parquet,
                           cameras.parquet, parameters.csv, rt_struct.parquet
  Carmel/processed/       same, + dFF/ (mean traces + peak-frame maps)
  Eenie/processed/        same as Austin
  Duluth/processed/       same as Carmel, + dFF/ has a Naive/Expert phase split
```

## What's not included

Raw widefield imaging (`.tif` stacks) and behavior camera video (`.avi`) are
excluded from all four subjects — several GB per subject, not practical for
an interactive dashboard and not needed for this level of analysis. The
imaging tab uses the trial-averaged ΔF/F summaries the original data already
contained (spatially-averaged time courses + peak-activity frame maps per
outcome category), not the raw movies.
