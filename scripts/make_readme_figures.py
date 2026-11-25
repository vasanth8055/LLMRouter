"""Generate reproducible PNG figures for README.md from saved results/v5/ files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "v5"
OUT_DIR = ROOT / "docs" / "images"

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
DARK = "#222222"
GRAY = "#7F8C8D"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_fig1_accuracy_ni(cv_refit: dict, phase5_ab: dict) -> Path:
    systems = cv_refit["systems"]
    platt = phase5_ab["attempt_a"]["metrics"]["platt"]

    entries = [
        ("Always\nDeepSeek", systems["deepseek"]["accuracy_pct"], systems["deepseek"]["accuracy_95ci_pct"], GRAY),
        ("Always\nGemini Flash", systems["flash"]["accuracy_pct"], systems["flash"]["accuracy_95ci_pct"], SKY),
        ("Always\nGemini Pro", systems["pro"]["accuracy_pct"], systems["pro"]["accuracy_95ci_pct"], DARK),
        ("Refit V2.1\n(Baseline CV)", systems["refit_v21_cv"]["accuracy_pct"], systems["refit_v21_cv"]["accuracy_95ci_pct"], ORANGE),
        ("V5 Platt\n(Default)", platt["accuracy"] * 100.0, [v * 100.0 for v in platt["accuracy_95ci"]], GREEN),
    ]

    labels = [e[0] for e in entries]
    means = [e[1] for e in entries]
    yerr_low = [e[1] - e[2][0] for e in entries]
    yerr_high = [e[2][1] - e[1] for e in entries]
    colors = [e[3] for e in entries]

    pro_acc = systems["pro"]["accuracy_pct"]
    ni_margin_line = pro_acc - 2.0

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
    x = np.arange(len(entries))
    bars = ax.bar(x, means, yerr=[yerr_low, yerr_high], capsize=5, width=0.56, color=colors, edgecolor="#111111", linewidth=0.7, error_kw={"ecolor": "#111111", "elinewidth": 1.3})

    ax.axhline(ni_margin_line, color=VERMILION, linestyle="--", linewidth=1.6, label=f"Pro - 2.00 pp NI Margin ({ni_margin_line:.2f}%)")
    ax.axhline(pro_acc, color=DARK, linestyle=":", linewidth=1.2, label=f"Always Gemini Pro ({pro_acc:.2f}%)")

    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            means[idx] + yerr_high[idx] + 0.35,
            f"{means[idx]:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%) [95% Bootstrap CI]")
    ax.set_ylim(73.5, 89.8)
    ax.set_title("MMLU-Pro Accuracy vs. 2.00 pp Non-Inferiority Margin\n(5-Fold Nested CV, N = 3,000)", fontsize=11.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92)
    fig.text(
        0.5,
        0.01,
        "V5 Platt paired non-inferiority vs Pro: diff = -1.13 pp, 95% CI lower bound = -1.70 pp (> -2.00 pp gate).",
        ha="center",
        fontsize=8.8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out = OUT_DIR / "fig1_accuracy_non_inferiority.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_fig2_frontier(phase5_final: dict) -> Path:
    floors = phase5_final["attempt_d_platt_frontier"]["floors"]
    taus = np.array([f["floor"] for f in floors])
    ret = np.array([f["retention_pct"] for f in floors])
    ret_lo = np.array([f["retention_95ci_pct"][0] for f in floors])
    ret_hi = np.array([f["retention_95ci_pct"][1] for f in floors])
    sav = np.array([f["savings_pct"] for f in floors])
    sav_lo = np.array([f["savings_95ci_pct"][0] for f in floors])
    sav_hi = np.array([f["savings_95ci_pct"][1] for f in floors])
    fb = np.array([f["fallback_pct"] for f in floors])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=160, sharex=True)

    # Left: Pro Retention
    ax1.plot(taus, ret, marker="o", color=BLUE, linewidth=2.0, label="Pro Quality Retention (%)")
    ax1.fill_between(taus, ret_lo, ret_hi, color=BLUE, alpha=0.18, label="95% Bootstrap CI")
    ax1.axhline(98.0, color=VERMILION, linestyle="--", linewidth=1.4, label="98.0% Pre-Registered Gate")
    ax1.axvline(0.85, color=GRAY, linestyle=":", linewidth=1.2, label="V5 Platt Default tau = 0.85")
    ax1.set_title("Pro Quality Retention vs. Floor")
    ax1.set_xlabel("Quality Floor (tau)")
    ax1.set_ylabel("Retention of Always-Pro (%)")
    ax1.set_ylim(92.5, 101.2)
    ax1.set_xticks(taus)
    ax1.set_xticklabels([f"{t:.3f}".rstrip("0").rstrip(".") if t != 0.8 else "0.80" for t in taus], rotation=25)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)

    # Right: Cost Savings & Fallback
    ax2.plot(taus, sav, marker="s", color=GREEN, linewidth=2.0, label="Offline Cost Savings vs. Pro (%)")
    ax2.fill_between(taus, sav_lo, sav_hi, color=GREEN, alpha=0.18, label="Savings 95% CI")
    ax2.plot(taus, fb, marker="^", color=ORANGE, linewidth=1.8, linestyle="-.", label="Pro Fallback Rate (%)")
    ax2.axvline(0.85, color=GRAY, linestyle=":", linewidth=1.2)
    ax2.set_title("Offline Cost Savings & Fallback vs. Floor")
    ax2.set_xlabel("Quality Floor (tau)")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_ylim(-3, 103)
    ax2.set_xticks(taus)
    ax2.set_xticklabels([f"{t:.3f}".rstrip("0").rstrip(".") if t != 0.8 else "0.80" for t in taus], rotation=25)
    ax2.legend(loc="upper left", fontsize=8.3, frameon=True, facecolor="white", framealpha=0.92)

    fig.suptitle(
        "Descriptive Cost-Quality Frontier: Platt Router across Fixed Floors (Nested CV, N = 3,000)",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        "Descriptive frontier only (results/v5/phase5_final.json); fixed-tau sweep without re-tuning inner-CV selection.",
        ha="center",
        fontsize=8.8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])

    out = OUT_DIR / "fig2_cost_quality_frontier.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_fig3_routing_share_fallback(cv_refit: dict, phase5_ab: dict) -> Path:
    v21 = cv_refit["systems"]["refit_v21_cv"]
    platt = phase5_ab["attempt_a"]["metrics"]["platt"]

    v21_share = v21["routing_distribution"]["percentages"]
    platt_share = platt["routing_share_pct"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.5), dpi=160)

    # Panel 1: Routing share stacked bar
    methods = ["Refit V2.1\n(Baseline)", "V5 Platt\n(Default)"]
    ds_vals = [v21_share["deepseek-v3-0324"], platt_share["deepseek-v3-0324"]]
    fl_vals = [v21_share["gemini-2.5-flash"], platt_share["gemini-2.5-flash"]]
    pr_vals = [v21_share["gemini-2.5-pro"], platt_share["gemini-2.5-pro"]]

    x = np.arange(2)
    w = 0.48
    ax1.bar(x, ds_vals, width=w, color=BLUE, label="DeepSeek V3")
    ax1.bar(x, fl_vals, width=w, bottom=ds_vals, color=SKY, label="Gemini 2.5 Flash")
    bottom_pr = [d + f for d, f in zip(ds_vals, fl_vals)]
    ax1.bar(x, pr_vals, width=w, bottom=bottom_pr, color=ORANGE, label="Gemini 2.5 Pro")

    for i in range(2):
        ax1.text(x[i], ds_vals[i] / 2, f"{ds_vals[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax1.text(x[i], ds_vals[i] + fl_vals[i] / 2, f"{fl_vals[i]:.1f}%", ha="center", va="center", color="#111111", fontweight="bold", fontsize=9)
        ax1.text(x[i], bottom_pr[i] + pr_vals[i] / 2, f"{pr_vals[i]:.1f}%", ha="center", va="center", color="#111111", fontweight="bold", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_ylim(0, 120)
    ax1.set_ylabel("Routed Prompts (%)")
    ax1.set_title("Model Routing Share", fontsize=11.2)
    ax1.legend(loc="upper center", ncol=3, fontsize=8.2, frameon=True)

    # Panel 2: Fallback Rate and Cost Savings
    metrics_x = np.arange(2)
    bw = 0.34
    v21_rates = [v21["fallback_rate_pct"], v21["cost_savings_vs_pro_pct"]]
    platt_rates = [platt["fallback_rate_pct"], platt["savings_vs_pro_pct"]]

    b1 = ax2.bar(metrics_x - bw / 2, v21_rates, width=bw, color=ORANGE, edgecolor="#111111", linewidth=0.6, label="Refit V2.1 (Baseline)")
    b2 = ax2.bar(metrics_x + bw / 2, platt_rates, width=bw, color=GREEN, edgecolor="#111111", linewidth=0.6, label="V5 Platt (Default)")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_xticks(metrics_x)
    ax2.set_xticklabels(["Pro Fallback Rate (-6.20 pp)", "Offline Savings (+2.08 pp)"])
    ax2.set_ylim(0, 62)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Fallback Rate & Offline Cost Savings", fontsize=11.2)
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.suptitle(
        "Routing Share, Fallback Rate, and Offline Cost Savings: Refit V2.1 vs. V5 Platt",
        fontsize=12.2,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = OUT_DIR / "fig3_routing_share_fallback.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_fig4_latency(phase5_ab: dict) -> Path:
    lat = phase5_ab["latencies"]
    models = [
        ("Refit V2.1\n(TF-IDF + LR)", lat["v21_tfidf_lr"]),
        ("V5 Platt\n(TF-IDF + LR + Platt)", lat["a_platt"]),
        ("Attempt 5b-1\n(all-MiniLM-L6-v2 + LR)", lat["b_minilm_only"]),
        ("Attempt 5b-2\n(MiniLM + TF-IDF + LR)", lat["b_minilm_plus_tfidf"]),
    ]

    labels = [m[0] for m in models]
    p50 = [m[1]["p50_ms"] for m in models]
    p95 = [m[1]["p95_ms"] for m in models]
    p99 = [m[1]["p99_ms"] for m in models]

    x = np.arange(len(models))
    bw = 0.25

    fig, ax = plt.subplots(figsize=(9.2, 4.7), dpi=160)
    b1 = ax.bar(x - bw, p50, width=bw, color=SKY, edgecolor="#111111", linewidth=0.6, label="p50 (ms)")
    b2 = ax.bar(x, p95, width=bw, color=BLUE, edgecolor="#111111", linewidth=0.6, label="p95 (ms)")
    b3 = ax.bar(x + bw, p99, width=bw, color=PURPLE, edgecolor="#111111", linewidth=0.6, label="p99 (ms)")

    ax.axhline(25.0, color=VERMILION, linestyle="--", linewidth=1.7, label="Pre-Registered p95 Budget (25.0 ms)")

    for bar in list(b1) + list(b2) + list(b3):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.2f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CPU Routing Decision Latency (ms)")
    ax.set_ylim(0, 35.5)
    ax.set_title("Single-Prompt CPU Routing Latency vs. 25 ms p95 Budget (N = 100 Warm Prompts)")
    ax.legend(loc="upper left", ncol=2, fontsize=9, frameon=True)
    fig.tight_layout()

    out = OUT_DIR / "fig4_latency_comparison.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_fig5_cross_domain(cross_domain: dict) -> Path:
    gpqa_lodo = cross_domain["lodo"]["gpqa"]["metrics"]["systems"]["lodo_text_cv"]
    gpqa_in = cross_domain["single_domain_nested_cv"]["gpqa"]["metrics"]["systems"]["indomain_cv"]
    lcb_lodo = cross_domain["lodo"]["livecodebench"]["metrics"]["systems"]["lodo_text_cv"]
    lcb_in = cross_domain["single_domain_nested_cv"]["livecodebench"]["metrics"]["systems"]["indomain_cv"]

    groups = [
        ("GPQA\nLODO\n(N=198)", gpqa_lodo),
        ("GPQA\nIn-Domain\n(N=198)", gpqa_in),
        ("LiveCodeBench\nLODO\n(N=1,055)", lcb_lodo),
        ("LiveCodeBench\nIn-Domain\n(N=1,055)", lcb_in),
    ]

    labels = [g[0] for g in groups]
    ret = [g[1]["pro_retention_pct"] for g in groups]
    sav = [g[1]["savings_vs_pro_pct"] for g in groups]
    fb = [g[1]["fallback_rate_pct"] for g in groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.6), dpi=160)
    x = np.arange(len(groups))

    # Left: Pro Retention
    bars_r = ax1.bar(x, ret, width=0.52, color=[VERMILION, BLUE, GRAY, GREEN], edgecolor="#111111", linewidth=0.6)
    ax1.axhline(98.0, color=VERMILION, linestyle="--", linewidth=1.4, label="98.0% Retention Reference")
    for bar, val in zip(bars_r, ret):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(92.0, 102.2)
    ax1.set_ylabel("Pro Quality Retention (%)")
    ax1.set_title("Cross-Domain Pro Retention")
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)

    # Right: Savings vs Pro and Fallback Rate
    bw = 0.36
    b_sav = ax2.bar(x - bw / 2, sav, width=bw, color=GREEN, edgecolor="#111111", linewidth=0.6, label="Offline Cost Savings vs. Pro (%)")
    b_fb = ax2.bar(x + bw / 2, fb, width=bw, color=ORANGE, edgecolor="#111111", linewidth=0.6, label="Pro Fallback Rate (%)")
    for bar in list(b_sav) + list(b_fb):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.4, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Cost Savings & Fallback Rate")
    ax2.legend(loc="upper left", fontsize=8.5, frameon=True)

    fig.suptitle(
        "Cross-Domain Transfer Failure vs. In-Domain Refit (GPQA & LiveCodeBench, inner-CV tau)",
        fontsize=12.0,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = OUT_DIR / "fig5_cross_domain.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    _apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cv_refit = json.loads((RESULTS_DIR / "cv_refit_v21.json").read_text(encoding="utf-8"))
    phase5_ab = json.loads((RESULTS_DIR / "phase5_ab.json").read_text(encoding="utf-8"))
    phase5_final = json.loads((RESULTS_DIR / "phase5_final.json").read_text(encoding="utf-8"))
    cross_domain = json.loads((RESULTS_DIR / "cross_domain.json").read_text(encoding="utf-8"))

    outputs = [
        plot_fig1_accuracy_ni(cv_refit, phase5_ab),
        plot_fig2_frontier(phase5_final),
        plot_fig3_routing_share_fallback(cv_refit, phase5_ab),
        plot_fig4_latency(phase5_ab),
        plot_fig5_cross_domain(cross_domain),
    ]
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
