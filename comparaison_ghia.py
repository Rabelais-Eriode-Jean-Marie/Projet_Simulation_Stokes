#!/usr/bin/env python3
"""
Comparaison profil Euler implicite vs Ghia et al. (1982) Re=100.

Prérequis :
    stokes_euler_profile.csv    (coupe x=0.5 : colonnes y, u1, u2)
    stokes_euler_profile_h.csv  (coupe y=0.5 : colonnes x, u1, u2)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# DONNÉES GHIA ET AL. 1982 — Re=100
# ============================================================

ghia_u = np.array([
    [1.0000,  1.00000],
    [0.9766,  0.84123],
    [0.9688,  0.78871],
    [0.9609,  0.73722],
    [0.9531,  0.68717],
    [0.8516,  0.23151],
    [0.7344,  0.00332],
    [0.6172, -0.13641],
    [0.5000, -0.20581],
    [0.4531, -0.21090],
    [0.2813, -0.15662],
    [0.1719, -0.10150],
    [0.1016, -0.06434],
    [0.0703, -0.04775],
    [0.0625, -0.04192],
    [0.0547, -0.03717],
    [0.0000,  0.00000],
])

ghia_v = np.array([
    [1.00000,  0.00000],
    [0.96880, -0.05906],
    [0.96090, -0.07391],
    [0.95310, -0.08864],
    [0.94530, -0.10313],
    [0.90630, -0.16914],
    [0.85940, -0.22445],
    [0.80470, -0.24533],
    [0.50000,  0.05454],
    [0.23440,  0.17527],
    [0.22660,  0.17507],
    [0.15630,  0.16077],
    [0.09380,  0.12317],
    [0.07810,  0.10890],
    [0.07030,  0.10091],
    [0.06250,  0.09233],
    [0.00000,  0.00000],
])

# ============================================================
# CHARGEMENT DES PROFILS FREEFEM
# ============================================================

dv = np.loadtxt("stokes_euler_profile.csv",   delimiter=",", skiprows=1)
dh = np.loadtxt("stokes_euler_profile_h.csv", delimiter=",", skiprows=1)

# dv : colonnes y, u1, u2   (coupe x=0.5)
# dh : colonnes x, u1, u2   (coupe y=0.5)

# ============================================================
# STYLE
# ============================================================

DARK  = "#0d1117"
PANEL = "#161b22"

plt.rcParams.update({
    "font.family"      : "monospace",
    "figure.facecolor" : DARK,
    "axes.facecolor"   : PANEL,
    "axes.edgecolor"   : "#30363d",
    "axes.labelcolor"  : "#e6edf3",
    "xtick.color"      : "#8b949e",
    "ytick.color"      : "#8b949e",
    "text.color"       : "#e6edf3",
    "grid.color"       : "#21262d",
    "legend.facecolor" : PANEL,
    "legend.edgecolor" : "#30363d",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(DARK)
fig.suptitle(
    "Euler implicite — Comparaison avec Ghia et al. (1982)\n"
    r"$\nu = 0.1$,  $T = 1.0$  vs  Ghia Re=100",
    fontsize=13, color="#e6edf3"
)

# ---- panneau gauche : u1(0.5, y) ----
ax = axes[0]
ax.plot(dv[:, 1], dv[:, 0],
        "-", color="#58a6ff", lw=2,
        label="Euler implicite (Stokes, $\\nu=0.1$)")
ax.plot(ghia_u[:, 1], ghia_u[:, 0],
        "wo", ms=6, mew=1.5,
        label="Ghia et al. 1982, Re=100")
ax.axvline(0, color="#555", lw=0.7, ls=":")
ax.set_xlabel(r"$u_1\,(0.5,\,y)$", fontsize=12)
ax.set_ylabel(r"$y$",              fontsize=12)
ax.set_title(r"Profil $u_1$ — coupe verticale $x = 0.5$", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# écart RMS
u1_pts = np.interp(ghia_u[:, 0], dv[:, 0], dv[:, 1])
rms_u  = np.sqrt(np.mean((u1_pts - ghia_u[:, 1])**2))
ax.text(0.04, 0.04, f"Écart RMS : {rms_u:.5f}",
        transform=ax.transAxes, fontsize=8, color="#8b949e")

# ---- panneau droit : u2(x, 0.5) ----
ax2 = axes[1]
ax2.plot(dh[:, 0], dh[:, 2],
         "-", color="#58a6ff", lw=2,
         label="Euler implicite (Stokes, $\\nu=0.1$)")
ax2.plot(ghia_v[:, 0], ghia_v[:, 1],
         "wo", ms=6, mew=1.5,
         label="Ghia et al. 1982, Re=100")
ax2.axhline(0, color="#555", lw=0.7, ls=":")
ax2.set_xlabel(r"$x$",               fontsize=12)
ax2.set_ylabel(r"$u_2\,(x,\,0.5)$", fontsize=12)
ax2.set_title(r"Profil $u_2$ — coupe horizontale $y = 0.5$", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# écart RMS
u2_pts = np.interp(ghia_v[:, 0], dh[:, 0], dh[:, 2])
rms_v  = np.sqrt(np.mean((u2_pts - ghia_v[:, 1])**2))
ax2.text(0.04, 0.04, f"Écart RMS : {rms_v:.5f}",
         transform=ax2.transAxes, fontsize=8, color="#8b949e")

fig.tight_layout()
out = Path("comparaison_ghia_euler.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"Figure sauvegardée : {out}")
plt.show()

# ============================================================
# TABLEAU
# ============================================================

print("\n" + "="*55)
print("  u1(0.5, y) — Euler implicite vs Ghia Re=100")
print("="*55)
print(f"{'y':>8}  {'Ghia Re=100':>12}  {'Euler':>10}  {'écart':>8}")
print("-"*55)
for row in ghia_u:
    y_g, u_g = row
    u_e = float(np.interp(y_g, dv[:, 0], dv[:, 1]))
    print(f"{y_g:8.4f}  {u_g:12.5f}  {u_e:10.5f}  {abs(u_e-u_g):8.5f}")
print(f"\n  Écart RMS total : {rms_u:.5f}")
print("="*55)

print("\n" + "="*55)
print("  u2(x, 0.5) — Euler implicite vs Ghia Re=100")
print("="*55)
print(f"{'x':>8}  {'Ghia Re=100':>12}  {'Euler':>10}  {'écart':>8}")
print("-"*55)
for row in ghia_v:
    x_g, v_g = row
    v_e = float(np.interp(x_g, dh[:, 0], dh[:, 2]))
    print(f"{x_g:8.4f}  {v_g:12.5f}  {v_e:10.5f}  {abs(v_e-v_g):8.5f}")
print(f"\n  Écart RMS total : {rms_v:.5f}")
print("="*55)