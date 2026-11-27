"""Depth-band analysis of window sweeps: do early, middle and late windows behave differently?

Windows are binned by ``center_depth_fraction`` (window centre / depth) into three
bands. For each parameter budget (discarded blocks), band and timestep bucket the
table reports the lowest one-step ``eps_err`` of any looped config and of plain
removal, so the effect of position is separated from the effect of looping.

    python -m loop_compression.bands --output team_runs/main
    python -m loop_compression.bands --output team_runs/main --extra early --finalist-budgets 4

``--extra TAG`` merges ``window_sweep_TAG.csv`` (e.g. from
``sensitivity windows --a-min 0 --tag TAG``) into ``window_sweep.csv``.
Outputs go to ``<output>/profile/sensitivity/bands/``: ``band_table.csv``,
``band_summary.csv`` (best config per budget and band), ``band_heatmap.png`` and
``../shortlist_bands.json`` plus ``../shortlist_bands/*.json``. The shortlist holds
one looped config per band with its same-window removal and runs through
``python -m loop_compression.drive validate --shortlist .../shortlist_bands.json``
for validation LPIPS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .policy import bucket_columns

BANDS = (("early", 0.0, 1 / 3), ("middle", 1 / 3, 2 / 3), ("late", 2 / 3, 1.0 + 1e-9))


def assign_band(center_fraction):
    for name, low, high in BANDS:
        if low <= center_fraction < high:
            return name
    raise ValueError(f"centre fraction outside [0, 1]: {center_fraction}")


def load_sweeps(sens_dir, extras=()):
    """Concatenate the base sweep with tagged extra sweeps; config ids are unique per setting."""
    sens_dir = Path(sens_dir)
    frames = [pd.read_csv(sens_dir / "window_sweep.csv")]
    for tag in extras:
        frames.append(pd.read_csv(sens_dir / f"window_sweep_{tag}.csv"))
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("config_id")
    frame = frame[frame.iteration == "block"].copy()
    frame["band"] = frame.center_depth_fraction.map(assign_band)
    return frame


def band_table(frame):
    """Best eps_err per (discard, band, bucket) for looped configs and for removal."""
    columns = bucket_columns(frame)
    rows = []
    for (discard, band), sub in frame.groupby(["discard", "band"]):
        loops = sub[sub.kind == "loop"]
        removals = sub[sub.kind == "removal"]
        if loops.empty or removals.empty:
            continue
        for bucket, column in enumerate(columns):
            loop_best = loops.loc[loops[column].idxmin()]
            removal_best = removals.loc[removals[column].idxmin()]
            rows.append(dict(
                discard=int(discard), band=band, bucket=bucket, windows=int(removals[["first", "last"]]
                                                                        .drop_duplicates().shape[0]),
                loop_eps_err=float(loop_best[column]), removal_eps_err=float(removal_best[column]),
                loop_over_removal=float(loop_best[column] / max(removal_best[column], 1e-12)),
                loop_window=f"{int(loop_best['first'])}-{int(loop_best['last'])}",
                loop_config=f"{loop_best.strategy} K={int(loop_best.K)} h={loop_best.h:g}"))
    return pd.DataFrame(rows)


def band_summary(frame):
    """Best looped config per (discard, band) by mean eps_err, with its same-window removal."""
    columns = bucket_columns(frame)
    rows = []
    for (discard, band), sub in frame.groupby(["discard", "band"]):
        loops = sub[sub.kind == "loop"]
        if loops.empty:
            continue
        best = loops.loc[loops[columns].mean(axis=1).idxmin()]
        same = sub[(sub.kind == "removal") & (sub["first"] == best["first"]) & (sub["last"] == best["last"])
                   & (sub.kept == best.kept)]
        removal_error = float(same[columns].mean(axis=1).iloc[0]) if len(same) else float("nan")
        rows.append(dict(discard=int(discard), band=band, config_id=best.config_id,
                         window=f"{int(best['first'])}-{int(best['last'])}", kept=best.kept,
                         centre=float(best.center_depth_fraction), strategy=best.strategy, K=int(best.K),
                         h=float(best.h), beta=float(best.beta),
                         loop_eps_err=float(best[columns].mean()), same_window_removal_eps_err=removal_error,
                         flops_ratio=float(best.flops_ratio), settings=best.settings))
    return pd.DataFrame(rows)


def finalists(summary, budgets):
    """Shortlist entries (loop + same-window removal) for the best config of each band."""
    entries = []
    for _, row in summary[summary.discard.isin(budgets)].iterrows():
        settings = json.loads(row.settings)
        base = f"d{row.discard}_{row.band}"
        entries.append(dict(name=f"{base}_loop", role="loop", budget=int(row.discard),
                            eps_err=float(row.loop_eps_err), settings=settings))
        removal = dict(settings, K=1, strategy="naive", h=1.0, beta=0.0)
        error = float(row.same_window_removal_eps_err)
        entries.append(dict(name=f"{base}_removal_same_window", role="removal_same_window",
                            budget=int(row.discard), eps_err=error if np.isfinite(error) else None,
                            settings=removal))
    return entries


def plot_bands(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    budgets = sorted(table.discard.unique())
    names = [b[0] for b in BANDS]
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.6 * len(budgets), 3.4), squeeze=False,
                             constrained_layout=True)
    for ax, budget in zip(axes[0], budgets):
        sub = table[table.discard == budget]
        loop = sub.pivot(index="band", columns="bucket", values="loop_eps_err").reindex(names)
        ratio = sub.pivot(index="band", columns="bucket", values="loop_over_removal").reindex(names)
        image = ax.imshow(np.log10(loop.to_numpy(dtype=float)), aspect="auto", origin="upper")
        for r in range(loop.shape[0]):
            for c in range(loop.shape[1]):
                value = ratio.to_numpy(dtype=float)[r, c]
                if np.isfinite(value):
                    ax.text(c, r, f"{value:.2f}x", ha="center", va="center", fontsize=7, color="white")
        ax.set_xticks(range(loop.shape[1]), [f"b{b}" for b in loop.columns])
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("timestep bucket (b0 = noisiest)")
        ax.set_title(f"{int(budget)} discarded: log10 best loop eps_err\n(text: loop / removal)", fontsize=8)
        fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extra", nargs="*", default=[], help="tags of extra window_sweep_<tag>.csv files")
    parser.add_argument("--finalist-budgets", type=int, nargs="+", default=[4])
    args = parser.parse_args(argv)
    sens = Path(args.output) / "profile" / "sensitivity"
    frame = load_sweeps(sens, args.extra)
    out = sens / "bands"
    out.mkdir(exist_ok=True)
    table, summary = band_table(frame), band_summary(frame)
    table.to_csv(out / "band_table.csv", index=False)
    summary.drop(columns="settings").to_csv(out / "band_summary.csv", index=False)
    plot_bands(table, out / "band_heatmap.png")
    entries = finalists(summary, args.finalist_budgets)
    if entries:
        from .sensitivity import write_shortlist
        write_shortlist(entries, sens / "shortlist_bands")
    print(summary.drop(columns=["settings", "config_id", "kept"]).round(4).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
