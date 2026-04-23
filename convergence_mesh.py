#!/usr/bin/env python3

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EDP_FILES = {
    "euler": Path("stokes_implicite.edp"),
    "cn":    Path("stokes_cn.edp"),
    "bdf2":  Path("stokes_bdf2.edp"),
}

LABELS = {
    "euler": "Euler implicite (ordre 1)",
    "cn":    "Crank-Nicolson (ordre 2)",
    "bdf2":  "BDF2 (ordre 2)",
}

COLORS  = {"euler": "#58a6ff", "cn": "#f78166", "bdf2": "#3fb950"}
MARKERS = {"euler": "o",       "cn": "s",        "bdf2": "^"}

N_VALUES = [8, 12, 16, 24, 32, 40]

DARK  = "#0d1117"
PANEL = "#161b22"


def patch_parameter(source: str, name: str, value: str) -> str:
    pattern = rf"(\b(?:int|real)\s+{re.escape(name)}\s*=\s*)([^;]+)(;)"
    patched, count = re.subn(pattern, rf"\g<1>{value}\3", source, count=1)
    if count == 0:
        raise ValueError(f"Parametre '{name}' introuvable.")
    return patched


def parse_l2_norm(output: str) -> float:
    match = re.search(r"\|\|u\|\|_L2\s*=\s*([0-9eE+\-.]+)", output)
    if not match:
        raise ValueError("Impossible de parser la norme finale ||u||_L2.")
    return float(match.group(1))


def run_case(source: str, n_value: int, edp_tmp: Path) -> float:
    patched = patch_parameter(source, "Nmesh", str(n_value))  # tes fichiers utilisent Nmesh
    patched = re.sub(r'\bsquare\s*\(\s*\d+\s*,\s*\d+\s*\)',
                     f'square({n_value}, {n_value})', patched)
    patched = patched.replace("wait=true", "wait=false")
    patched = patched.replace("wait=1", "wait=0")
    patched = re.sub(r'\bplot\s*\([^;]*\)\s*;', '', patched, flags=re.DOTALL)
    edp_tmp.write_text(patched, encoding="utf-8")

    result = subprocess.run(
        ["FreeFem++", str(edp_tmp)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        tail = (result.stdout + "\n" + result.stderr)[-1500:]
        raise RuntimeError(f"FreeFem++ a echoue pour N={n_value}.\n{tail}")

    return parse_l2_norm(result.stdout)


def make_plot(all_results: dict) -> None:
    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": DARK,
        "axes.facecolor": PANEL,
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#8b949e",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "text.color": "#e6edf3",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#21262d",
        "legend.facecolor": PANEL,
        "legend.edgecolor": "#30363d",
    })

    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.patch.set_facecolor(DARK)

    for schema, (h_values, norms) in all_results.items():
        ax.loglog(h_values, norms,
                  marker=MARKERS[schema],
                  color=COLORS[schema],
                  linewidth=2, markersize=7,
                  markerfacecolor=DARK,
                  markeredgewidth=1.8,
                  label=LABELS[schema])
        for h, val in zip(h_values, norms):
            ax.annotate(f"{val:.3e}", (h, val),
                        textcoords="offset points",
                        xytext=(4, 6), fontsize=7,
                        color=COLORS[schema])

    ax.set_title("Sensibilite au maillage - norme finale")
    ax.set_xlabel("h ~= 1/N")
    ax.set_ylabel(r"$||u(T)||_{L^2}$")
    ax.legend()
    ax.grid(True, which="both")

    out = Path("sensibilite_maillage.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    print(f"\nGraphe sauvegarde : {out}")
    plt.show()


def main() -> int:
    all_results = {}

    for schema, edp_file in EDP_FILES.items():
        if not edp_file.exists():
            print(f"[SKIP] Fichier introuvable : {edp_file}")
            continue

        edp_tmp = Path(f"_tmp_{schema}_sens.edp")
        source  = edp_file.read_text(encoding="utf-8")
        results = []

        print("=" * 68)
        print(f"  {LABELS[schema]}")
        print(f"  N testes : {N_VALUES}")
        print("=" * 68)

        try:
            for n_value in N_VALUES:
                h_value = 1.0 / n_value
                print(f"\n-- N = {n_value:>3d}  (h ~= {h_value:.5f})")
                l2_norm = run_case(source, n_value, edp_tmp)
                results.append((h_value, l2_norm))
                print(f"   ||u(T)||_L2 = {l2_norm:.6e}")
        finally:
            if edp_tmp.exists():
                edp_tmp.unlink()

        h_values = np.array([r[0] for r in results])
        norms    = np.array([r[1] for r in results])

        coeffs = np.polyfit(np.log(h_values), np.log(norms), 1)
        print(f"\nPente log-log : {coeffs[0]:+.4f}")

        all_results[schema] = (h_values, norms)

    if all_results:
        make_plot(all_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())