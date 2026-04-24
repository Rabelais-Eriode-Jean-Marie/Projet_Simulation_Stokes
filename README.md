# Stokes Instationnaire — Cavité Entraînée

Projet numérique MACS2  
Schémas temporels pour les équations de Stokes instationnaires appliqués au problème de la cavité entraînée (lid-driven cavity).

---

## Structure du projet

```
.
├── stokes_implicite.edp        # Schéma Euler implicite (ordre 1)
├── stokes_cn.edp               # Schéma Crank-Nicolson (ordre 2)
├── stokes_bdf2.edp             # Schéma BDF2 (ordre 2)
│
├── plot_stokes.py              # Visualisation du champ final (6 panneaux)
├── convergence_ordre.py        # Convergence temporelle en dt (loglog)
├── convergence_mesh.py         # Convergence spatiale en h (loglog)
├── sensibilite_maillage.py     # Sensibilité au maillage — norme ||u(T)||_L2
├── sensibilite_nu.py           # Sensibilité à la viscosité — norme ||u(T)||_L2
├── comparaison_ghia.py         # Comparaison profils vitesse vs Ghia et al. (1982)
│
└── README.md
```

---

## Problème physique

On résout les équations de Stokes instationnaires sur le carré unité $\Omega = [0,1]^2$ :

$$\partial_t u - \nu \Delta u + \nabla p = f \quad \text{dans } \Omega \times (0, T]$$
$$\nabla \cdot u = 0 \quad \text{dans } \Omega \times (0, T]$$

avec les conditions aux limites de cavité entraînée :
- $u = (1, 0)$ sur le bord supérieur (bord 3)
- $u = 0$ sur les trois autres bords

et la condition initiale $u_0 = 0$.

---

## Schémas temporels implémentés

### Euler implicite (ordre 1)

$$\frac{u^{n+1} - u^n}{\Delta t} - \nu \Delta u^{n+1} + \nabla p^{n+1} = f^{n+1}$$

Schéma inconditionnellement stable, premier ordre en temps. Implémenté dans `stokes_implicite.edp`.

### Crank-Nicolson (ordre 2)

$$\frac{u^{n+1} - u^n}{\Delta t} - \frac{\nu}{2}(\Delta u^{n+1} + \Delta u^n) + \nabla p^{n+1} = f^{n+1}$$

Schéma d'ordre 2 en temps, inconditionnellement stable. Implémenté dans `stokes_cn.edp`.  
**Note** : l'ordre 2 théorique est atteint uniquement pour des données initiales régulières. Sur le problème de cavité entraînée (condition initiale incompatible avec les CL), l'ordre observé est proche de 1 en pratique — comportement attendu et documenté.

### BDF2 (ordre 2)

$$\frac{3u^{n+1} - 4u^n + u^{n-1}}{2\Delta t} - \nu \Delta u^{n+1} + \nabla p^{n+1} = f^{n+1}$$

Schéma de différentiation rétrograde d'ordre 2, A-stable. Le premier pas est amorcé par un Euler implicite. Implémenté dans `stokes_bdf2.edp`.

---

## Discrétisation spatiale

Éléments finis mixtes Taylor-Hood **P2/P1** :
- Vitesse : éléments $P_2$ (continus, quadratiques)
- Pression : éléments $P_1$ (continus, linéaires)

La paire P2/P1 satisfait la condition inf-sup de Babuška-Brezzi, garantissant la stabilité de la discrétisation en pression.

---

## Dépendances

- [FreeFEM++](https://freefem.org/) ≥ 4.10
- Python ≥ 3.10
- `numpy`, `matplotlib`, `pandas`

Installation des dépendances Python :

```bash
pip install numpy matplotlib pandas
```

---

## Usage

### Visualisation du champ final

```bash
python plot_stokes.py euler
python plot_stokes.py cn
python plot_stokes.py bdf2
```

Produit une figure 6 panneaux : $u_1$, $u_2$, pression, norme de vitesse, lignes de courant, profil vertical $u_1(x=0.5, y)$.

### Convergence temporelle

```bash
python convergence_ordre.py euler
python convergence_ordre.py cn
python convergence_ordre.py bdf2
python convergence_ordre.py all
```

Fait varier $\Delta t$ à maillage fixe et trace $\|u_h - u_{ref}\|_{L^2}$ en loglog.  
La solution de référence est calculée à $\Delta t_{ref}$ très petit et mise en cache (`cache_ref_*.npy`).  
Pour forcer le recalcul : supprimer les fichiers `cache_ref_*.npy`.

### Convergence spatiale

```bash
python convergence_mesh.py euler
python convergence_mesh.py cn
python convergence_mesh.py bdf2
python convergence_mesh.py all
```

Fait varier $N$ à $\Delta t$ fixe et trace l'erreur en loglog.

### Sensibilité au maillage

```bash
python sensibilite_maillage.py
```

Trace $\|u(T)\|_{L^2}$ en fonction de $h = 1/N$ pour les trois schémas.

### Sensibilité à la viscosité

```bash
python sensibilite_nu.py
```

Trace $\|u(T)\|_{L^2}$ en fonction de $\nu$ à maillage fixe pour les trois schémas.

### Comparaison avec Ghia et al. (1982)

Ajouter d'abord les blocs d'export dans chaque `.edp` (voir ci-dessous), puis :

```bash
python comparaison_ghia.py
```

Produit une figure deux panneaux et un tableau d'écarts RMS entre les profils de vitesse et les données de référence de Ghia et al. (1982) pour Re=100.

#### Blocs d'export à ajouter dans chaque `.edp`

```freefem
// Export profil vertical x=0.5
{
    ofstream fp1("stokes_euler_profile.csv");
    fp1 << "y,u1,u2\n";
    for(int i = 0; i <= 300; i++){
        real yy = i * 1.0 / 300;
        fp1 << yy << "," << u(0.5, yy) << "," << v(0.5, yy) << "\n";
    }
}
// Export profil horizontal y=0.5
{
    ofstream fp2("stokes_euler_profile_h.csv");
    fp2 << "x,u1,u2\n";
    for(int i = 0; i <= 300; i++){
        real xx = i * 1.0 / 300;
        fp2 << xx << "," << u(xx, 0.5) << "," << v(xx, 0.5) << "\n";
    }
}
```

Adapter le nom du CSV (`stokes_euler` → `stokes_cn` / `stokes_bdf2`) selon le schéma.

---

## Résultats

### Ordres de convergence observés

| Schéma | Ordre théorique | Ordre observé (cavité) |
|--------|----------------|------------------------|
| Euler implicite | 1 | ~1.0 |
| Crank-Nicolson  | 2 | ~1.0–1.5 (singularité CI) |
| BDF2            | 2 | ~2.0 |

L'ordre réduit de CN sur la cavité est dû à l'incompatibilité entre la condition initiale nulle et la condition au bord $u=1$ sur le bord supérieur — cette singularité initiale dégrade l'ordre global. Sur un problème à solution régulière (solution manufacturée), CN retrouve son ordre 2 théorique.

### Sensibilité physique

- La norme $\|u(T)\|_{L^2}$ est quasi-indépendante du maillage pour $N \geq 16$ — les trois schémas convergent vers la même solution physique.
- La norme décroît avec $\nu$ : à faible viscosité, la dissipation est moindre et l'écoulement est plus intense.

---

## Paramètres par défaut

| Paramètre | Valeur |
|-----------|--------|
| $\nu$ (viscosité) | 0.1 |
| $T$ (temps final) | 1.0 |
| $\Delta t$ | 0.05–0.1 selon le schéma |
| $N$ (maillage) | 20 |
| Éléments finis | Taylor-Hood P2/P1 |

---

## Auteurs

**Rabelais RIOLA**  
**Eriode HOUNTONGBE**  
**Jean-Marie AGOUNDO**  

