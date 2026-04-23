#!/usr/bin/env python3
"""
Etude de sensibilite en viscosite pour les schemas de Stokes instationnaire.
Fait varier nu a maillage fixe et releve la norme finale ||u||_L2.
"""

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

N_FIXED   = 25
NU_VALUES = [1.0, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]

DARK  = "#0d1117"
PANEL = "#161b22"

# noms de la viscosite dans chaque fichier
NU_PARAM = {
    "euler": "nu",
    "cn":    "viscosite",
    "bdf2":  "viscosite",
}


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


def run_case(source: str, nu_value: float, schema: str, edp_tmp: Path) -> float:
    patched = patch_parameter(source, "Nmesh", str(N_FIXED))
    patched = patch_parameter(patched, NU_PARAM[schema], f"{nu_value:.8e}")
    patched = patched.replace("wait=true", "wait=false")
    patched = patched.replace("wait=1",    "wait=0")
    patched = re.sub(r'plot\s*\([^)]*\)\s*;', '', patched, flags=re.DOTALL)
    edp_tmp.write_text(patched, encoding="utf-8")

    result = subprocess.run(
        ["FreeFem++", str(edp_tmp)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        tail = (result.stdout + "\n" + result.stderr)[-1500:]
        raise RuntimeError(f"FreeFem++ a echoue pour nu={nu_value:.3e}.\n{tail}")

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

    for schema, (nu_values, norms) in all_results.items():
        ax.loglog(nu_values, norms,
                  marker=MARKERS[schema],
                  color=COLORS[schema],
                  linewidth=2, markersize=7,
                  markerfacecolor=DARK,
                  markeredgewidth=1.8,
                  label=LABELS[schema])
        for nu_value, val in zip(nu_values, norms):
            ax.annotate(f"{val:.3e}", (nu_value, val),
                        textcoords="offset points",
                        xytext=(4, 6), fontsize=8,
                        color=COLORS[schema])

    ax.invert_xaxis()
    ax.set_title(f"Sensibilite en viscosite - N = {N_FIXED}")
    ax.set_xlabel("nu")
    ax.set_ylabel(r"$||u(T)||_{L^2}$")
    ax.legend()
    ax.grid(True, which="both")

    out = Path("sensibilite_nu.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    print(f"\nGraphe sauvegarde : {out}")
    plt.show()


def main() -> int:
    all_results = {}

    for schema, edp_file in EDP_FILES.items():
        if not edp_file.exists():
            print(f"[SKIP] Fichier introuvable : {edp_file}")
            continue

        edp_tmp = Path(f"_tmp_{schema}_nu.edp")
        source  = edp_file.read_text(encoding="utf-8")
        results = []

        print("=" * 68)
        print(f"  {LABELS[schema]}")
        print(f"  N fixe    : {N_FIXED}")
        print(f"  nu testes : {NU_VALUES}")
        print("=" * 68)

        try:
            for nu_value in NU_VALUES:
                print(f"\n-- nu = {nu_value:.3e}")
                l2_norm = run_case(source, nu_value, schema, edp_tmp)
                results.append((nu_value, l2_norm))
                print(f"   ||u(T)||_L2 = {l2_norm:.6e}")
        finally:
            if edp_tmp.exists():
                edp_tmp.unlink()

        nu_values = np.array([r[0] for r in results], dtype=float)
        norms     = np.array([r[1] for r in results], dtype=float)

        print("\n" + "=" * 44)
        print(f"{'nu':>12} {'||u(T)||_L2':>18}")
        print("-" * 44)
        for nu_value, l2_norm in results:
            print(f"{nu_value:>12.3e} {l2_norm:>18.6e}")
        print("=" * 44)

        coeffs = np.polyfit(np.log(nu_values), np.log(norms), 1)
        print(f"Pente log-log indicative : {coeffs[0]:+.4f}")

        all_results[schema] = (nu_values, norms)

    if all_results:
        make_plot(all_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())