"""
Plots for the empirical strategic manipulation analysis.
Reads manipulation_results_YS.csv and manipulation_results_RR.csv
from the current directory and saves PNGs alongside them.

Color palette mirrors the Plotly qualitative palette used in the notebook.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Files ────────────────────────────────────────────────────────────────────
YS_FILE = "manipulation_results_YS.csv"
RR_FILE = "manipulation_results_RR.csv"

# ─── Palette (Plotly qualitative, reordered to match notebook) ────────────────
# indices [0,1,2,4,3,5] of the Plotly palette
_PLOTLY = [
    "#636EFA", "#EF553B", "#00CC96",
    "#AB63FA", "#FFA15A", "#19D3F3",
]
PALETTE = [_PLOTLY[i] for i in [0, 1, 2, 4, 3, 5]]
FADED   = [c + "88" for c in PALETTE]  # ~53 % opacity

# Semantic aliases
COL_TRUTHFUL  = FADED[0]   # blue  – truthful utility
COL_GAIN      = FADED[1]   # red   – manipulation gain
COL_GAP       = FADED[2]   # green – remaining gap / no-gain
COL_HIGHLIGHT = PALETTE[1] # solid red for mean lines etc.

# ─── Total students per status in the reduced instance (from notebook) ─────────
# status 1=Freshman … 6=PhD
TOTAL_PER_STATUS = {1: 24, 2: 33, 3: 41, 4: 57, 5: 61, 6: 15}
STATUS_NAMES     = {1: "Freshman", 2: "Sophomore", 3: "Junior",
                    4: "Senior",   5: "Masters",   6: "PhD"}

# ─── Strategy grouping ────────────────────────────────────────────────────────
STRATEGY_ORDER  = ["boost_missed", "demote_assigned", "only_missed", "promote_single"]
STRATEGY_LABELS = ["H1\nboost_missed", "H3\ndemote_assigned",
                   "H2\nonly_missed",  "H4\npromote_single"]

DPI = 150


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_data():
    ys = pd.read_csv(YS_FILE)
    rr = pd.read_csv(RR_FILE)
    for df in [ys, rr]:
        df["strategy_type"] = df["strategy"].apply(
            lambda s: "promote_single" if s.startswith("promote_single") else s
        )
        df["usw_change"] = df["total_usw_new"] - df["total_usw_old"]
    return ys, rr


def per_student_summary(df):
    """One row per student: regret_before, max_gain, old_utility, status."""
    base = (
        df.groupby("student")
        .agg(
            regret_before=("regret_before", "first"),
            old_utility=("old_utility", "first"),
            max_gain=("gain", "max"),
            status=("student_status", "first"),
        )
        .reset_index()
    )
    base["max_gain"] = base["max_gain"].clip(lower=0)  # cap at 0 if all losses
    base["remaining_gap"] = (base["regret_before"] - base["max_gain"]).clip(lower=0)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1  Stacked bar: truthful utility / manipulation gain / remaining gap
# (one bar per student, grouped by status — mirrors notebook cell-19)
# ─────────────────────────────────────────────────────────────────────────────

def fig_utility_stacked(ys, rr):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    status_order = [6, 5, 4, 3, 2, 1]  # PhD first (matches notebook sort)

    for ax, (df, title) in zip(axes, [(ys, "Yankee Swap"), (rr, "Round Robin")]):
        ps = per_student_summary(df)

        # Sort students: status descending, then by student index
        ps = ps.sort_values(["status", "student"], ascending=[False, True])
        x = np.arange(len(ps))

        ax.bar(x, ps["old_utility"],   color=COL_TRUTHFUL, label="Truthful utility")
        ax.bar(x, ps["max_gain"],      color=COL_GAIN,     label="Best manipulation gain",
               bottom=ps["old_utility"].values)
        ax.bar(x, ps["remaining_gap"], color=COL_GAP,      label="Remaining gap to OPT",
               bottom=(ps["old_utility"] + ps["max_gain"]).values)

        # Status dividers and labels
        boundaries = [0]
        for st in status_order:
            n = (ps["status"] == st).sum()
            boundaries.append(boundaries[-1] + n)
        for b in boundaries[1:-1]:
            ax.axvline(b - 0.5, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
        for i, st in enumerate(status_order):
            mid = (boundaries[i] + boundaries[i + 1]) / 2
            ax.text(mid, -0.04, STATUS_NAMES[st],
                    transform=ax.get_xaxis_transform(),
                    ha="center", fontsize=8)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Utility", fontsize=10)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    plt.suptitle("Truthful utility, manipulation gain, and remaining gap to OPT",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("fig_utility_stacked.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig_utility_stacked.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2  Manipulability by status — stacked horizontal bar
# (mirrors notebook cell-22)
# ─────────────────────────────────────────────────────────────────────────────

def fig_by_status(ys, rr):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, (df, title) in zip(axes, [(ys, "Yankee Swap"), (rr, "Round Robin")]):
        statuses = sorted(TOTAL_PER_STATUS.keys())
        labels   = [STATUS_NAMES[s] for s in statuses]

        happy_counts    = []
        can_gain_counts = []
        no_gain_counts  = []

        for st in statuses:
            total   = TOTAL_PER_STATUS[st]
            sub     = df[df["student_status"] == st]
            tested  = sub["student"].nunique()
            profitable = sub[sub["gain"] > 0]["student"].nunique()

            happy_counts.append(total - tested)
            can_gain_counts.append(profitable)
            no_gain_counts.append(tested - profitable)

        y = np.arange(len(statuses))
        h = np.array(happy_counts)
        g = np.array(can_gain_counts)
        n = np.array(no_gain_counts)

        ax.barh(y, h, color=COL_TRUTHFUL, label="Happy (YS = OPT)")
        ax.barh(y, g, left=h,     color=COL_GAIN, label="Can gain via misreport")
        ax.barh(y, n, left=h + g, color=COL_GAP,  label="No gain found")

        # Annotate counts inside bars
        for i in range(len(statuses)):
            if h[i] > 0:
                ax.text(h[i] / 2, i, str(h[i]),
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            if g[i] > 0:
                ax.text(h[i] + g[i] / 2, i, str(g[i]),
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            if n[i] > 0:
                ax.text(h[i] + g[i] + n[i] / 2, i, str(n[i]),
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Number of students", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    handles = [
        mpatches.Patch(color=COL_TRUTHFUL, label="Happy (YS = OPT)"),
        mpatches.Patch(color=COL_GAIN,     label="Can gain via misreport"),
        mpatches.Patch(color=COL_GAP,      label="No gain found"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))
    plt.suptitle("Manipulability by student status", fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("fig_by_status.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig_by_status.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3  Regret distribution (mirrors notebook cell-23)
# ─────────────────────────────────────────────────────────────────────────────

def fig_regret_distribution(ys, rr):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, (df, title) in zip(axes, [(ys, "Yankee Swap"), (rr, "Round Robin")]):
        regrets = df.groupby("student")["regret_before"].first()
        ax.hist(regrets, bins=20, color=FADED[1], edgecolor="white")
        ax.axvline(regrets.mean(), color=COL_HIGHLIGHT, linewidth=1.5,
                   linestyle="--", label=f"Mean = {regrets.mean():.1f}")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Regret = OPT − truthful utility", fontsize=10)
        ax.set_ylabel("Number of students" if ax is axes[0] else "", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Distribution of regret (OPT − truthful allocation utility)", fontsize=13)
    plt.tight_layout()
    plt.savefig("fig_regret_distribution.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig_regret_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4  Best manipulation gain distribution (mirrors notebook cell-24)
# ─────────────────────────────────────────────────────────────────────────────

def fig_gain_hist(ys, rr):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, (df, title) in zip(axes, [(ys, "Yankee Swap"), (rr, "Round Robin")]):
        ps = per_student_summary(df)
        pos = ps[ps["max_gain"] > 0]["max_gain"]
        ax.hist(pos, bins=15, color=FADED[1], edgecolor="white")
        ax.axvline(pos.mean(), color=COL_HIGHLIGHT, linewidth=1.5,
                   linestyle="--", label=f"Mean = {pos.mean():.1f}")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Best manipulation gain", fontsize=10)
        ax.set_ylabel("Number of students" if ax is axes[0] else "", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Distribution of best manipulation gain\n(among students with at least one profitable strategy)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("fig_gain_hist.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig_gain_hist.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5  Regret vs best gain scatter (mirrors notebook cell-25)
# ─────────────────────────────────────────────────────────────────────────────

def fig_regret_vs_gain(ys, rr):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, (df, title) in zip(axes, [(ys, "Yankee Swap"), (rr, "Round Robin")]):
        ps = per_student_summary(df)
        can_gain = ps[ps["max_gain"] > 0]
        no_gain  = ps[ps["max_gain"] == 0]

        ax.scatter(no_gain["regret_before"],  no_gain["max_gain"],
                   color=COL_GAP,  alpha=0.5, label="No gain", s=30)
        ax.scatter(can_gain["regret_before"], can_gain["max_gain"],
                   color=COL_GAIN, alpha=0.7, label="Profitable", s=30)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Regret (OPT − truthful utility)", fontsize=10)
        ax.set_ylabel("Best manipulation gain" if ax is axes[0] else "", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Regret vs. best manipulation gain", fontsize=13)
    plt.tight_layout()
    plt.savefig("fig_regret_vs_gain.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig_regret_vs_gain.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6  Fraction of regret recovered (mirrors notebook cell-26)
# ─────────────────────────────────────────────────────────────────────────────

def fig_regret_recovered(ys, rr):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, (df, title) in zip(axes, [(ys, "Yankee Swap"), (rr, "Round Robin")]):
        ps = per_student_summary(df)
        frac = ps["max_gain"] / ps["regret_before"]
        ax.hist(frac, bins=20, color=FADED[1], edgecolor="white")
        ax.axvline(frac.mean(), color=COL_HIGHLIGHT, linewidth=1.5,
                   linestyle="--", label=f"Mean = {frac.mean():.2f}")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Fraction of regret recoverable", fontsize=10)
        ax.set_ylabel("Number of students" if ax is axes[0] else "", fontsize=10)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("How much of their regret can students recover via manipulation?", fontsize=12)
    plt.tight_layout()
    plt.savefig("fig_regret_recovered.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig_regret_recovered.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7  Gain distribution per heuristic — violin plots
# ─────────────────────────────────────────────────────────────────────────────

def fig_gain_distribution(ys, rr):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, (df, title) in zip(axes, [(ys, "Yankee Swap"), (rr, "Round Robin")]):
        data = [df[df["strategy_type"] == st]["gain"].values for st in STRATEGY_ORDER]
        positions = np.arange(len(STRATEGY_ORDER))
        vp = ax.violinplot(data, positions=positions, showmedians=True, widths=0.7)
        for patch, c in zip(vp["bodies"], FADED[:4]):
            patch.set_facecolor(c)
        for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
            vp[key].set_color("black")
        ax.axhline(0, color=COL_HIGHLIGHT, linestyle="--", linewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(STRATEGY_LABELS, fontsize=9)
        ax.set_ylabel("Utility gain" if ax is axes[0] else "", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Distribution of utility gains per heuristic", fontsize=13)
    plt.tight_layout()
    plt.savefig("fig_gain_distribution.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig_gain_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8  Outcome breakdown — profitable / neutral / harmful (stacked bar)
# ─────────────────────────────────────────────────────────────────────────────

def fig_outcome_breakdown(ys, rr):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (df, title) in zip(axes, [(ys, "Yankee Swap"), (rr, "Round Robin")]):
        profs, neutr, harms = [], [], []
        for st in STRATEGY_ORDER:
            sub = df[df["strategy_type"] == st]["gain"]
            n = len(sub)
            profs.append(100 * (sub > 0).sum() / n)
            neutr.append(100 * (sub == 0).sum() / n)
            harms.append(100 * (sub < 0).sum() / n)

        x = np.arange(len(STRATEGY_ORDER))
        ax.bar(x, profs, color=PALETTE[2], label="Profitable (gain > 0)")
        ax.bar(x, neutr, color="#AAAAAA",  label="Neutral (gain = 0)", bottom=profs)
        bottoms = [p + n for p, n in zip(profs, neutr)]
        ax.bar(x, harms, color=PALETTE[1], label="Harmful (gain < 0)", bottom=bottoms)

        for i, v in enumerate(profs):
            if v > 3:
                ax.text(i, v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        for i, (b, v) in enumerate(zip(bottoms, harms)):
            if v > 5:
                ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGY_LABELS, fontsize=9)
        ax.set_ylabel("% of attempts" if ax is axes[0] else "", fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

    handles = [
        mpatches.Patch(color=PALETTE[2], label="Profitable (gain > 0)"),
        mpatches.Patch(color="#AAAAAA",  label="Neutral (gain = 0)"),
        mpatches.Patch(color=PALETTE[1], label="Harmful (gain < 0)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))
    plt.suptitle("Outcome breakdown per heuristic", fontsize=13)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig("fig_outcome_breakdown.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig_outcome_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9  Social welfare impact (mirrors fig_usw_impact)
# ─────────────────────────────────────────────────────────────────────────────

def fig_usw_impact(ys, rr):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (df, title) in zip(axes, [(ys, "Yankee Swap"), (rr, "Round Robin")]):
        usw_all  = df["usw_change"].values
        usw_prof = df[df["gain"] > 0]["usw_change"].values

        lo   = min(usw_all.min(), -5)
        hi   = max(usw_all.max(),  5)
        bins = np.arange(lo, hi + 2, 2)

        ax.hist(usw_all,  bins=bins, alpha=0.55, color=PALETTE[0],
                label="All strategies",     density=True, edgecolor="white")
        ax.hist(usw_prof, bins=bins, alpha=0.75, color=PALETTE[1],
                label="Profitable only", density=True, edgecolor="white")

        ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
        ax.axvline(usw_all.mean(),  color=PALETTE[0], linewidth=1.5,
                   label=f"Mean all ({usw_all.mean():.1f})")
        if len(usw_prof):
            ax.axvline(usw_prof.mean(), color=PALETTE[1], linewidth=1.5,
                       label=f"Mean profitable ({usw_prof.mean():.1f})")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Change in total social welfare (ΔUSW)", fontsize=10)
        ax.set_ylabel("Density" if ax is axes[0] else "", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Social welfare impact of manipulations", fontsize=13)
    plt.tight_layout()
    plt.savefig("fig_usw_impact.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig_usw_impact.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ys, rr = load_data()

    fig_utility_stacked(ys, rr)
    fig_by_status(ys, rr)
    fig_regret_distribution(ys, rr)
    fig_gain_hist(ys, rr)
    fig_regret_vs_gain(ys, rr)
    fig_regret_recovered(ys, rr)
    fig_gain_distribution(ys, rr)
    fig_outcome_breakdown(ys, rr)
    fig_usw_impact(ys, rr)
