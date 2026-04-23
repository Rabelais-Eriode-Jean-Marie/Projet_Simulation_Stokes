#!/usr/bin/env python3
"""
Visualisation pour les schemas de Stokes instationnaire.
Lance FreeFem++, exporte un echantillonnage regulier du champ final,
puis affiche vitesse, pression et lignes de courant.

Usage :
    python plot_stokes.py euler
    python plot_stokes.py cn
    python plot_stokes.py bdf2
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


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

# noms des variables vitesse et pression dans chaque fichier
VAR_NAMES = {
    "euler": ("u", "v", "p"),
    "cn":    ("u", "v", "p"),
    "bdf2":  ("u", "v", "p"),
}

SAMPLE_POINTS = 81


def build_export_block(csv_path: Path, u1: str, u2: str, vp: str) -> str:
    return rf"""
// ============================================================
// Export Python
// ============================================================
{{
    ofstream csvpy("{csv_path.as_posix()}");
    csvpy << "x,y,u1,u2,p\n";
    for (int i = 0; i < {SAMPLE_POINTS}; i++) {{
        real xx = i * 1.0 / ({SAMPLE_POINTS} - 1);
        for (int j = 0; j < {SAMPLE_POINTS}; j++) {{
            real yy = j * 1.0 / ({SAMPLE_POINTS} - 1);
            csvpy << xx << "," << yy << ","
                  << {u1}(xx,yy) << ","
                  << {u2}(xx,yy) << ","
                  << {vp}(xx,yy) << "\n";
        }}
    }}
}}
"""


def parse_l2_norm(output: str) -> float | None:
    match = re.search(r"\|\|u\|\|_L2\s*=\s*([0-9eE+\-.]+)", output)
    return float(match.group(1)) if match else None


def run_freefem(schema: str) -> tuple[Path, float | None]:
    edp_file = EDP_FILES[schema]
    if not edp_file.exists():
        raise FileNotFoundError(f"Fichier EDP introuvable : {edp_file}")

    u1, u2, vp = VAR_NAMES[schema]
    csv_file = Path(f"stokes_{schema}_data.csv")
    edp_tmp  = Path(f"_tmp_plot_{schema}.edp")

    source = edp_file.read_text(encoding="utf-8")
    source = source.replace("wait=true", "wait=false")
    source = source.replace("wait=1",    "wait=0")
    import re as _re
    source = _re.sub(r'\bplot\s*\([^;]*\)\s*;', '', source, flags=_re.DOTALL)
    source += "\n" + build_export_block(csv_file, u1, u2, vp)

    edp_tmp.write_text(source, encoding="utf-8")

    print(f"[{schema.upper()}] Lancement FreeFem++ ...")
    result = subprocess.run(
        ["FreeFem++", str(edp_tmp)],
        capture_output=True,
        text=True,
    )

    edp_tmp.unlink(missing_ok=True)

    if result.stdout:
        print(result.stdout.strip())

    if result.returncode != 0:
        tail = (result.stdout + "\n" + result.stderr)[-2000:]
        raise RuntimeError(f"FreeFem++ a echoue.\n{tail}")

    if not csv_file.exists():
        raise FileNotFoundError(f"Fichier de donnees introuvable : {csv_file}")

    return csv_file, parse_l2_norm(result.stdout)


def load_data(csv_file: Path):
    df = pd.read_csv(csv_file)
    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    df = df.sort_values(["x", "y"]).reset_index(drop=True)

    nx = len(xs)
    ny = len(ys)

    x_grid = df["x"].to_numpy().reshape(nx, ny)
    y_grid = df["y"].to_numpy().reshape(nx, ny)
    u_grid = df["u1"].to_numpy().reshape(nx, ny)
    v_grid = df["u2"].to_numpy().reshape(nx, ny)
    p_grid = df["p"].to_numpy().reshape(nx, ny)
    return x_grid, y_grid, u_grid, v_grid, p_grid


def make_figure(schema: str, x_grid, y_grid, u_grid, v_grid, p_grid,
                l2_norm: float | None) -> None:
    speed = np.sqrt(u_grid**2 + v_grid**2)

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#0f1117",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "#aaaaaa",
        "ytick.color": "#aaaaaa",
        "axes.edgecolor": "#333333",
    })

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#0f1117")
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
                  left=0.06, right=0.97, top=0.92, bottom=0.06)

    title = f"Stokes instationnaire - {LABELS[schema]}"
    if l2_norm is not None:
        title += f"  |  ||u(T)||_L2 = {l2_norm:.4e}"
    fig.suptitle(title, fontsize=15, color="white", y=0.97)

    pmax   = max(abs(float(p_grid.min())), abs(float(p_grid.max())), 1e-12)
    norm_p = mcolors.TwoSlopeNorm(vmin=-pmax, vcenter=0.0, vmax=pmax)

    ax1 = fig.add_subplot(gs[0, 0])
    c1  = ax1.contourf(x_grid, y_grid, u_grid, levels=40, cmap="RdBu_r")
    fig.colorbar(c1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title("u1(x,y)")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_aspect("equal")

    ax2 = fig.add_subplot(gs[0, 1])
    c2  = ax2.contourf(x_grid, y_grid, v_grid, levels=40, cmap="RdBu_r")
    fig.colorbar(c2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title("u2(x,y)")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_aspect("equal")

    ax3 = fig.add_subplot(gs[0, 2])
    c3  = ax3.contourf(x_grid, y_grid, p_grid, levels=40, cmap="seismic", norm=norm_p)
    fig.colorbar(c3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_title("Pression p(x,y)")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_aspect("equal")

    ax4 = fig.add_subplot(gs[1, 0])
    c4  = ax4.contourf(x_grid, y_grid, speed, levels=40, cmap="inferno")
    fig.colorbar(c4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.set_title(r"Norme de vitesse $|u|$")
    ax4.set_xlabel("x"); ax4.set_ylabel("y"); ax4.set_aspect("equal")

    ax5 = fig.add_subplot(gs[1, 1])
    xs  = x_grid[:, 0]
    ys  = y_grid[0, :]
    ax5.streamplot(xs, ys, u_grid.T, v_grid.T,
                   color=speed.T, cmap="plasma",
                   linewidth=1.2, density=1.3, arrowsize=0.9)
    ax5.set_title("Lignes de courant")
    ax5.set_xlabel("x"); ax5.set_ylabel("y")
    ax5.set_xlim(0, 1); ax5.set_ylim(0, 1); ax5.set_aspect("equal")

    ax6  = fig.add_subplot(gs[1, 2])
    ix   = int(np.argmin(np.abs(xs - 0.5)))
    ax6.plot(u_grid[ix, :], ys, color="#58a6ff", linewidth=2)
    ax6.set_title("Profil vertical de u1 en x = 0.5")
    ax6.set_xlabel("u1(0.5, y)"); ax6.set_ylabel("y")
    ax6.grid(True, alpha=0.25)

    out = Path(f"stokes_{schema}_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"\nFigure sauvegardee : {out}")
    plt.show()


def main() -> int:
    valid = list(EDP_FILES.keys())

    if len(sys.argv) < 2:
        print(f"Usage : python plot_stokes.py [{' | '.join(valid)}]")
        return 1

    schema = sys.argv[1].lower()
    if schema not in valid:
        print(f"Choix invalide : '{schema}'.  Choisir parmi : {', '.join(valid)}")
        return 1

    try:
        csv_file, l2_norm = run_freefem(schema)
        x_grid, y_grid, u_grid, v_grid, p_grid = load_data(csv_file)
        make_figure(schema, x_grid, y_grid, u_grid, v_grid, p_grid, l2_norm)
    except Exception as exc:
        print(f"Erreur : {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())