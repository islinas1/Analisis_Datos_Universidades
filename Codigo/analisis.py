import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

plt.style.use("dark_background")
CYAN, PINK, PURPLE = "#06b6d4", "#f472b6", "#a78bfa"
GREEN, ORANGE, RED = "#34d399", "#fb923c", "#ef4444"
YELLOW, BLUE = "#facc15", "#38bdf8"

COLS = [
    "MATRICULADOS HOMBRES", "MATRICULADOS MUJERES",
    "NUEVOS HOMBRES", "NUEVOS MUJERES",
    "TITULADOS HOMBRES", "TITULADOS MUJERES",
]
CORTO = {"MATRICULADOS HOMBRES": "Mat.H", "MATRICULADOS MUJERES": "Mat.M",
         "NUEVOS HOMBRES": "Nuev.H", "NUEVOS MUJERES": "Nuev.M",
         "TITULADOS HOMBRES": "Tit.H", "TITULADOS MUJERES": "Tit.M"}

AREA_CORTO = {
    "ARQUITECTURA, URBANISMO Y ARTE": "Arquitectura",
    "CIENCIAS AGRICOLAS PECUARIAS Y FORESTALES": "Agropecuarias",
    "CIENCIAS BASICAS Y NATURALES": "C.Básicas",
    "CIENCIAS DE LA COMUNICACION": "Comunicación",
    "CIENCIAS DE LA EDUCACION Y HUMANIDADES": "Educación",
    "CIENCIAS DE LA SALUD": "Salud",
    "CIENCIAS ECONOMICAS FINANCIERAS Y ADMINISTRATIVAS": "Económicas",
    "CIENCIAS SOCIALES": "C.Sociales",
    "INGENIERIA Y TECNOLOGIA": "Ingeniería",
    "OTROS OFICIOS": "Otros",
}

# Carpeta de salida para gráficos (relativa al script)
DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DIR, "..", "informe")
os.makedirs(OUT, exist_ok=True)


def guardar(fig, nombre):
    """Guarda figura en la carpeta informe."""
    ruta = os.path.join(OUT, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {nombre}")


# ================================================================
# 1. CARGA
# ================================================================
print("=" * 60)
print("1. CARGA DEL DATASET")
print("=" * 60)

df = pd.read_csv(os.path.join(DIR, "universities_bolivia_dataset.csv"))
print(f"  Filas: {len(df):,}  |  Columnas: {df.shape[1]}")
print(f"  Años: {df['AÑO'].min()} – {df['AÑO'].max()}")
print(f"  Departamentos: {sorted(df['CIUDAD'].unique())}")
print(f"  Tipo: {list(df['TIPO DE EDUCACIÓN'].unique())}")


# ================================================================
# 2. DATOS FALTANTES
# ================================================================
print("\n" + "=" * 60)
print("2. DATOS FALTANTES")
print("=" * 60)

# Filas con todo en cero = sin datos reportados → NaN
todo_cero = df[COLS].sum(axis=1) == 0
print(f"  Filas sin datos (todos ceros): {todo_cero.sum():,} ({todo_cero.mean()*100:.1f}%)")

df_work = df.copy()
for c in COLS:
    df_work.loc[todo_cero, c] = np.nan

# Resumen antes de imputar
print(f"\n  {'Variable':<12} {'Faltantes':>9} {'Media':>9} {'Mediana':>9}")
print(f"  {'-'*12} {'-'*9} {'-'*9} {'-'*9}")
for c in COLS:
    na = df_work[c].isna().sum()
    me = df_work[c].mean()
    md = df_work[c].median()
    print(f"  {CORTO[c]:<12} {na:>9,} {me:>9.1f} {md:>9.1f}")

# Imputar con mediana por grupo
for c in COLS:
    mediana_grupo = df_work.groupby(["ÁREA", "TIPO DE EDUCACIÓN"])[c].transform("median")
    df_work[c] = df_work[c].fillna(mediana_grupo).fillna(df_work[c].median())

print(f"\n  Método: mediana por grupo (ÁREA + TIPO DE EDUCACIÓN)")
print(f"  NaN restantes: {df_work[COLS].isna().sum().sum()}")


# ================================================================
# 3. OUTLIERS
# ================================================================
print("\n" + "=" * 60)
print("3. DETECCIÓN DE OUTLIERS")
print("=" * 60)

# 3a. Método IQR
print("\n  ── Método IQR (1.5 × IQR) ──")
print(f"  {'Var':<9} {'Q1':>7} {'Q3':>7} {'IQR':>7} {'Límite':>8} {'Outliers':>8} {'%':>6}")
print(f"  {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*6}")

iqr_res = {}
for c in COLS:
    v = df_work[c][df_work[c] > 0]
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    limite = q3 + 1.5 * iqr
    n_out = (v > limite).sum()
    iqr_res[c] = {"q1": q1, "q3": q3, "iqr": iqr, "limite": limite, "n": n_out}
    print(f"  {CORTO[c]:<9} {q1:>7.0f} {q3:>7.0f} {iqr:>7.0f} {limite:>8.0f} {n_out:>8,} {n_out/len(v)*100:>5.1f}%")

# 3b. Winsorización (5%-95%)
print("\n  ── Winsorización (P5 – P95) ──")
print(f"  {'Var':<9} {'Media':>8} {'→ Win':>8} {'Std':>8} {'→ Win':>8} {'Afect':>7}")
print(f"  {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")

win_res = {}
for c in COLS:
    v = df_work[c][df_work[c] > 0]
    w = stats.mstats.winsorize(v, limits=[0.05, 0.05])
    n_af = int(((v < np.percentile(v, 5)) | (v > np.percentile(v, 95))).sum())
    win_res[c] = {"orig_m": v.mean(), "win_m": w.mean(),
                  "orig_s": v.std(), "win_s": w.std(), "n": n_af}
    print(f"  {CORTO[c]:<9} {v.mean():>8.1f} {w.mean():>8.1f} {v.std():>8.1f} {w.std():>8.1f} {n_af:>7,}")

# Comparativa
print("\n  ── Comparativa ──")
print(f"  {'Var':<9} {'IQR':>8} {'Winsor':>8}")
for c in COLS:
    print(f"  {CORTO[c]:<9} {iqr_res[c]['n']:>8,} {win_res[c]['n']:>8,}")
print("  → IQR es más estricto en todas las variables.")


# ================================================================
# 4. MÉTRICAS DESCRIPTIVAS
# ================================================================
print("\n" + "=" * 60)
print("4. MÉTRICAS DESCRIPTIVAS")
print("=" * 60)

resumen = []
for c in COLS:
    v = df_work[c]
    resumen.append({
        "Var": CORTO[c], "N": len(v), "Media": v.mean(),
        "Mediana": v.median(), "Std": v.std(),
        "Mín": v.min(), "Máx": v.max(),
        "Asimetría": v.skew(), "Curtosis": v.kurtosis(),
    })
print(pd.DataFrame(resumen).to_string(index=False, float_format="{:.1f}".format))

print("\n  ── Por Tipo de Educación ──")
print(df_work.groupby("TIPO DE EDUCACIÓN")[COLS].mean().round(1).rename(columns=CORTO))

print("\n  ── Por Departamento (Matriculados) ──")
cm = df_work.groupby("CIUDAD")[["MATRICULADOS HOMBRES", "MATRICULADOS MUJERES"]].mean().round(1)
cm.columns = ["Hombres", "Mujeres"]
cm["Total"] = cm.sum(axis=1)
print(cm.sort_values("Total", ascending=False))


# ================================================================
# 5. GRÁFICOS
# ================================================================
print("\n" + "=" * 60)
print("5. GENERANDO GRÁFICOS")
print("=" * 60)

colores_box = [CYAN, PINK, PURPLE, GREEN, ORANGE, YELLOW]

# ── Fig 1: Box Plots ──
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Diagramas de Caja", fontsize=16, fontweight="bold")
for i, c in enumerate(COLS):
    ax = axes[i // 3][i % 3]
    data = df_work[c][df_work[c] > 0]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor=colores_box[i] + "40", edgecolor=colores_box[i]),
                    medianprops=dict(color=PINK, linewidth=2),
                    flierprops=dict(marker=".", color=RED, markersize=2, alpha=0.3))
    ax.set_title(CORTO[c], color=colores_box[i], fontweight="bold")
    ax.set_xlabel(f"Outliers IQR: {iqr_res[c]['n']:,}", fontsize=8, color=RED)
    ax.grid(alpha=0.2)
plt.tight_layout()
guardar(fig, "fig1_boxplots.png")

# ── Fig 2: Dispersión H vs M ──
fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle("Dispersión: Matriculados Hombres vs Mujeres", fontsize=14, fontweight="bold")
mask = (df_work["MATRICULADOS HOMBRES"] > 0) & (df_work["MATRICULADOS MUJERES"] > 0)
sample = df_work[mask].sample(min(500, mask.sum()), random_state=42)
areas = sorted(sample["ÁREA"].unique())
for j, area in enumerate(areas):
    sub = sample[sample["ÁREA"] == area]
    ax.scatter(sub["MATRICULADOS HOMBRES"], sub["MATRICULADOS MUJERES"],
               s=20, alpha=0.5, label=AREA_CORTO.get(area, area),
               color=colores_box[j % len(colores_box)])
lim = max(sample["MATRICULADOS HOMBRES"].max(), sample["MATRICULADOS MUJERES"].max())
ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.4, label="Paridad H=M")
ax.set_xlabel("Hombres")
ax.set_ylabel("Mujeres")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.2)
plt.tight_layout()
guardar(fig, "fig2_dispersion.png")

# ── Fig 3: Barras por Área ──
fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle("Matriculados Promedio por Área", fontsize=14, fontweight="bold")
am = df_work.groupby("ÁREA")[["MATRICULADOS HOMBRES", "MATRICULADOS MUJERES"]].mean()
am["short"] = am.index.map(AREA_CORTO)
am = am.sort_values("MATRICULADOS HOMBRES")
y = range(len(am))
ax.barh([i - 0.18 for i in y], am["MATRICULADOS HOMBRES"], 0.35, label="Hombres", color=CYAN)
ax.barh([i + 0.18 for i in y], am["MATRICULADOS MUJERES"], 0.35, label="Mujeres", color=PINK)
ax.set_yticks(list(y))
ax.set_yticklabels(am["short"])
ax.legend()
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
guardar(fig, "fig3_barras_area.png")

# ── Fig 4: Barras por Ciudad ──
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("Matriculados Promedio por Departamento", fontsize=14, fontweight="bold")
cm2 = df_work.groupby("CIUDAD")[["MATRICULADOS HOMBRES", "MATRICULADOS MUJERES"]].mean()
cm2 = cm2.sort_values("MATRICULADOS HOMBRES")
y = range(len(cm2))
ax.barh([i - 0.18 for i in y], cm2["MATRICULADOS HOMBRES"], 0.35, label="Hombres", color=CYAN)
ax.barh([i + 0.18 for i in y], cm2["MATRICULADOS MUJERES"], 0.35, label="Mujeres", color=PINK)
ax.set_yticks(list(y))
ax.set_yticklabels(cm2.index)
ax.legend()
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
guardar(fig, "fig4_barras_ciudad.png")

# ── Fig 5: Evolución temporal ──
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("Evolución de Matrícula Total por Año", fontsize=14, fontweight="bold")
yt = df_work.groupby("AÑO")[["MATRICULADOS HOMBRES", "MATRICULADOS MUJERES"]].sum() / 1000
ax.plot(yt.index, yt["MATRICULADOS HOMBRES"], "-o", color=CYAN, lw=2.5, ms=6, label="Hombres")
ax.plot(yt.index, yt["MATRICULADOS MUJERES"], "-o", color=PINK, lw=2.5, ms=6, label="Mujeres")
ax.fill_between(yt.index, yt["MATRICULADOS HOMBRES"], yt["MATRICULADOS MUJERES"],
                alpha=0.15, color=PURPLE)
ax.axvline(2012, ls="--", color=YELLOW, alpha=0.5)
ax.annotate("2012: cruce de género", xy=(2012, yt.iloc[6].mean()), fontsize=9,
            color=YELLOW, ha="center")
ax.set_xlabel("Año")
ax.set_ylabel("Matriculados (miles)")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
guardar(fig, "fig5_evolucion.png")

# ── Fig 6: IQR vs Winsorización ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Comparativa: IQR vs Winsorización", fontsize=14, fontweight="bold")
labels = [CORTO[c] for c in COLS]
x = np.arange(len(labels))

ax1.bar(x - 0.18, [iqr_res[c]["n"] for c in COLS], 0.35, label="IQR", color=RED)
ax1.bar(x + 0.18, [win_res[c]["n"] for c in COLS], 0.35, label="Winsor", color=ORANGE)
ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=30)
ax1.set_title("Cantidad de outliers detectados", fontsize=11)
ax1.legend(); ax1.grid(axis="y", alpha=0.2)

ax2.bar(x - 0.18, [win_res[c]["orig_s"] for c in COLS], 0.35, label="Std Original", color=PURPLE)
ax2.bar(x + 0.18, [float(win_res[c]["win_s"]) for c in COLS], 0.35, label="Std Winsorizada", color=GREEN)
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30)
ax2.set_title("Reducción de Desv. Estándar", fontsize=11)
ax2.legend(); ax2.grid(axis="y", alpha=0.2)

plt.tight_layout()
guardar(fig, "fig6_comparativa_outliers.png")

# ── Fig 7: Media vs Mediana ──
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Media vs Mediana (justifica usar mediana para imputar)", fontsize=13, fontweight="bold")
medias = [df_work[c].mean() for c in COLS]
medianas = [df_work[c].median() for c in COLS]
ax.bar(x - 0.18, medias, 0.35, label="Media", color=PINK)
ax.bar(x + 0.18, medianas, 0.35, label="Mediana", color=CYAN)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.legend(); ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
guardar(fig, "fig7_media_vs_mediana.png")

# ── Fig 8: Pública vs Privada ──
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("Promedio: Pública vs Privada", fontsize=14, fontweight="bold")
tm = df_work.groupby("TIPO DE EDUCACIÓN")[COLS].mean()
ax.bar(x - 0.18, tm.loc["PUBLICA"], 0.35, label="Pública", color=CYAN)
ax.bar(x + 0.18, tm.loc["PRIVADA"], 0.35, label="Privada", color=ORANGE)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.legend(); ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
guardar(fig, "fig8_publica_vs_privada.png")

print("\n" + "=" * 60)
print("LISTO — 8 gráficos guardados en /informe")
print("=" * 60)
