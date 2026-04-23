#!/usr/bin/env python3
"""
Convergence temporelle (en dt) pour les schemas de Stokes instationnaire.
Inspire du style du prof : une figure par schema, loglog, pentes de reference.

Usage :
    python convergence_ordre.py euler
    python convergence_ordre.py cn
    python convergence_ordre.py bdf2
    python convergence_ordre.py all
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG
# ============================================================

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

DT_VALUES = {
    "euler": [0.05,  0.02,  0.01,  0.005],
    "cn":    [0.05,  0.02,  0.01,  0.005],
    "bdf2":  [0.02,  0.01,  0.005, 0.002],
}

DT_REF = {
    "euler": 5e-4,
    "cn":    5e-4,
    "bdf2":  5e-4,
}

N_FIXED = {
    "euler": 40,
    "cn":    80,
    "bdf2":  40,
}

T_FIXED = 0.5
NPTS    = 300

VAR_NAMES = {
    "euler": ("u", "v"),
    "cn":    ("u", "v"),
    "bdf2":  ("u", "v"),
}

DARK  = "#0d1117"
PANEL = "#161b22"

# ============================================================
# PATCH SOURCE
# ============================================================

def patch_source(src: str, N: int, dt: float) -> str:
    src = re.sub(r'\bsquare\s*\(\s*\d+\s*,\s*\d+\s*\)',
                 f'square({N}, {N})', src)
    src = re.sub(r'(\breal\s+dt\s*=\s*)([^;]+)(;)',
                 rf'\g<1>{dt:.10e}\3', src, count=1)
    src = re.sub(r'(\breal\s+T\s*=\s*)([^;]+)(;)',
                 rf'\g<1>{T_FIXED}\3', src, count=1)
    src = re.sub(r'(\breal\s+Tfinal\s*=\s*)([^;]+)(;)',
                 rf'\g<1>{T_FIXED}\3', src, count=1)
    M_val = max(1, round(T_FIXED / dt))
    src = re.sub(r'(\bint\s+M\s*=\s*)([^;]+)(;)',
                 rf'\g<1>{M_val}\3', src, count=1)
    src = src.replace("wait=true", "wait=false")
    src = src.replace("wait=1",    "wait=0")
    src = re.sub(r'\bplot\s*\([^;]*\)\s*;', '', src, flags=re.DOTALL)
    src = re.sub(r'cout\s*<<[^;]+<<\s*endl\s*;', '', src)
    return src

# ============================================================
# EXPORT CSV
# ============================================================

def export_block(csv_path: Path, u1: str, u2: str) -> str:
    return f"""
{{
    ofstream fexp("{csv_path.as_posix()}");
    fexp << "y,u1,u2\\n";
    for (int ii = 0; ii <= {NPTS}; ii++) {{
        real yy = ii * 1.0 / {NPTS};
        fexp << yy << "," << {u1}(0.5, yy) << "," << {u2}(0.5, yy) << "\\n";
    }}
}}
"""

# ============================================================
# RUN FREEFEM
# ============================================================

def run_case(schema: str, source: str, N: int, dt: float) -> np.ndarray:
    u1, u2 = VAR_NAMES[schema]
    M_val  = max(1, round(T_FIXED / dt))

    tag     = f"{schema}_N{N}_dt{dt:.6f}".replace(".", "p")
    tmp_edp = Path(f"tmp_{tag}.edp")
    csv     = Path(f"tmp_{tag}.csv")

    code = patch_source(source, N, dt)
    code += export_block(csv, u1, u2)
    tmp_edp.write_text(code, encoding="utf-8")

    print(f"    dt={dt:.5f}  M={M_val:5d} ...", end="", flush=True)

    res = subprocess.run(["FreeFem++", str(tmp_edp)],
                         capture_output=True, text=True)
    tmp_edp.unlink(missing_ok=True)

    if res.returncode != 0:
        print(" FAIL")
        print(res.stdout[-2000:])
        raise RuntimeError(f"FreeFem++ a echoue (schema={schema}, dt={dt})")

    if not csv.exists():
        raise RuntimeError(f"CSV non genere : {csv}")

    data = np.loadtxt(str(csv), delimiter=",", skiprows=1)
    csv.unlink(missing_ok=True)
    print("  OK")
    return data

# ============================================================
# ERREUR L2
# ============================================================

def compute_error(data: np.ndarray, ref: np.ndarray) -> float:
    y      = data[:, 0]
    u1_ref = np.interp(y, ref[:, 0], ref[:, 1])
    u2_ref = np.interp(y, ref[:, 0], ref[:, 2])
    dy     = y[1] - y[0]
    return float(np.sqrt(dy * np.sum(
        (data[:, 1] - u1_ref) ** 2 +
        (data[:, 2] - u2_ref) ** 2
    )))

# ============================================================
# PIPELINE PAR SCHEMA
# ============================================================

def run_schema(schema: str) -> list[tuple[float, float]]:
    edp_file = EDP_FILES[schema]
    if not edp_file.exists():
        raise FileNotFoundError(f"Fichier introuvable : {edp_file}")

    source  = edp_file.read_text(encoding="utf-8")
    dt_ref  = DT_REF[schema]
    dt_list = sorted(DT_VALUES[schema], reverse=True)
    N       = N_FIXED[schema]

    print("=" * 68)
    print(f"  {LABELS[schema]}")
    print(f"  N={N}  T={T_FIXED}  dt_ref={dt_ref:.1e}")
    print("=" * 68)

    # cache disque pour la reference
    cache = Path(f"cache_ref_{schema}.npy")
    if cache.exists():
        print(f"  [Reference dt={dt_ref:.1e}]  (cache)")
        ref = np.load(str(cache))
    else:
        print(f"  [Reference dt={dt_ref:.1e}]")
        ref = run_case(schema, source, N, dt_ref)
        np.save(str(cache), ref)

    print(f"\n  {'dt':>10}   {'erreur':>12}   {'ordre':>7}")
    print(f"  {'-'*10}   {'-'*12}   {'-'*7}")

    results  = []
    err_prev = None
    dt_prev  = None

    for dt in dt_list:
        data = run_case(schema, source, N, dt)
        err  = compute_error(data, ref)

        if err < 1e-14:
            print(f"  {dt:10.5f}   {err:12.4e}   {'~0':>7}")
            continue

        if err_prev is not None:
            ordre = np.log(err_prev / err) / np.log(dt_prev / dt)
            print(f"  {dt:10.5f}   {err:12.4e}   {ordre:7.3f}")
        else:
            print(f"  {dt:10.5f}   {err:12.4e}   {'—':>7}")

        results.append((dt, err))
        err_prev = err
        dt_prev  = dt

    return results

# ============================================================
# PLOT PAR SCHEMA (style prof)
# ============================================================

def make_plot(schema: str, results: list[tuple[float, float]]) -> None:
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

    rs     = sorted(results)
    dt_arr = np.array([r[0] for r in rs])
    e_arr  = np.array([r[1] for r in rs])

    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.patch.set_facecolor(DARK)

    ax.loglog(dt_arr, e_arr,
              marker=MARKERS[schema],
              color=COLORS[schema],
              linewidth=2, markersize=7,
              markerfacecolor=DARK,
              markeredgewidth=1.8,
              label=LABELS[schema])

    for dt, err in zip(dt_arr, e_arr):
        ax.annotate(f"{err:.2e}", (dt, err),
                    textcoords="offset points",
                    xytext=(5, 6), fontsize=8,
                    color=COLORS[schema])

    # pentes ancrées sur le plus grand dt
    dt0, e0 = dt_arr[-1], e_arr[-1]
    dt_span = np.array([dt_arr[0], dt_arr[-1]])
    ax.loglog(dt_span, e0*(dt_span/dt0)**1, "w--", lw=1.2, alpha=0.6, label="ordre 1")
    ax.loglog(dt_span, e0*(dt_span/dt0)**2, "w:",  lw=1.2, alpha=0.6, label="ordre 2")

    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$\|u_h - u_{ref}\|_{L^2}$")
    ax.set_title(f"Convergence temporelle — {LABELS[schema]}\n"
                 f"N={N_FIXED[schema]}, T={T_FIXED}, ref dt={DT_REF[schema]:.1e}")
    ax.legend(fontsize=9)
    ax.grid(True, which="both")

    out = Path(f"convergence_ordre_{schema}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    print(f"\nFigure sauvegardee : {out}")
    plt.show()

# ============================================================
# MAIN
# ============================================================

def main() -> int:
    valid = list(EDP_FILES.keys()) + ["all"]

    if len(sys.argv) < 2:
        print(f"Usage : python convergence_ordre.py [{' | '.join(valid)}]")
        return 0

    arg = sys.argv[1].lower()
    if arg not in valid:
        print(f"Choix invalide : '{arg}'.  Choisir parmi : {', '.join(valid)}")
        return 1

    schemas = list(EDP_FILES.keys()) if arg == "all" else [arg]

    for schema in schemas:
        try:
            results = run_schema(schema)
            if results:
                make_plot(schema, results)
        except FileNotFoundError as exc:
            print(f"\n[SKIP] {exc}")
        except RuntimeError as exc:
            print(f"\n[ERREUR] {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())