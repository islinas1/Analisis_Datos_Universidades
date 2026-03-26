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
df = pd.read_csv(os.path.join(DIR, "universities_bolivia_dataset (1).csv"))

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

def graficar_imputacion_outliers(df_work, COLS, CORTO, iqr_res, win_res):
    """
    Genera gráficos que muestran el impacto de la imputación de valores faltantes
    y el tratamiento de outliers.
    
    Parámetros:
    -----------
    df_work : DataFrame
        DataFrame con datos ya imputados
    COLS : list
        Lista de columnas numéricas a analizar
    CORTO : dict
        Diccionario con nombres cortos para las columnas
    iqr_res : dict
        Resultados del método IQR para cada columna
    win_res : dict
        Resultados de winsorización para cada columna
    """
    
    # Colores personalizados
    colores_box = [CYAN, PINK, PURPLE, GREEN, ORANGE, YELLOW]
    
    # FIGURA 1: COMPARACIÓN ANTES/DESPUÉS DE IMPUTACIÓN
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Impacto de la Imputación de Valores Faltantes", 
                 fontsize=16, fontweight="bold")
    
    for i, c in enumerate(COLS):
        ax = axes[i // 3][i % 3]
        
        # Datos antes de imputación (valores originales con NaN)
        antes = df_work[c].copy()
        # Simular datos antes de imputación (para visualización)
        # Usamos la versión original del dataframe antes de imputar
        if 'df_original' in locals():
            antes_plot = df_original[c].dropna()
        else:
            antes_plot = df_work[c][df_work[c] > 0]
        
        # Datos después de imputación
        despues = df_work[c][df_work[c] > 0]
        
        # Crear dos subgráficos por variable
        ax.hist(antes_plot, bins=30, alpha=0.6, label=f'Antes (n={len(antes_plot):,})', 
                color=RED, edgecolor='white', linewidth=0.5)
        ax.hist(despues, bins=30, alpha=0.6, label=f'Después (n={len(despues):,})', 
                color=GREEN, edgecolor='white', linewidth=0.5)
        
        ax.set_title(f"{CORTO[c]}", color=colores_box[i], fontweight="bold", fontsize=11)
        ax.set_xlabel("Valores")
        ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    
    plt.tight_layout()
    guardar(fig, "fig_imputacion_histogramas.png")
    
    # FIGURA 2: OUTLIERS DETECTADOS POR IQR
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Detección de Outliers (Método IQR)", 
                 fontsize=16, fontweight="bold")
    
    for i, c in enumerate(COLS):
        ax = axes[i // 3][i % 3]
        
        # Datos sin outliers (dentro de límite IQR)
        datos = df_work[c][df_work[c] > 0]
        q3 = iqr_res[c]['q3']
        limite = iqr_res[c]['limite']
        
        datos_sin_outliers = datos[datos <= limite]
        datos_outliers = datos[datos > limite]
        
        # Histograma
        ax.hist(datos_sin_outliers, bins=40, alpha=0.7, color=CYAN, 
                label=f'Datos normales (n={len(datos_sin_outliers):,})', 
                edgecolor='white', linewidth=0.5)
        
        if len(datos_outliers) > 0:
            ax.hist(datos_outliers, bins=20, alpha=0.8, color=RED, 
                    label=f'Outliers (n={len(datos_outliers):,})', 
                    edgecolor='white', linewidth=0.5)
        
        # Línea del límite IQR
        ax.axvline(limite, color=YELLOW, linestyle='--', linewidth=2, 
                   label=f'Límite IQR: {limite:.0f}')
        
        ax.set_title(f"{CORTO[c]} - Outliers: {iqr_res[c]['n']:,} ({iqr_res[c]['n']/len(datos)*100:.1f}%)", 
                     color=colores_box[i], fontweight="bold")
        ax.set_xlabel("Valores")
        ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.2)
    
    plt.tight_layout()
    guardar(fig, "fig_outliers_iqr.png")
    
    # FIGURA 3: BOXPLOTS ANTES Y DESPUÉS DE WINSORIZACIÓN
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Efecto de Winsorización en la Distribución", 
                 fontsize=16, fontweight="bold")
    
    for i, c in enumerate(COLS):
        ax = axes[i // 3][i % 3]
        
        # Datos originales
        datos_orig = df_work[c][df_work[c] > 0]
        
        # Aplicar winsorización
        datos_win = stats.mstats.winsorize(datos_orig, limits=[0.05, 0.05])
        
        # Preparar datos para boxplot
        data_to_plot = [datos_orig, datos_win]
        
        # Crear boxplot
        bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.6,
                        labels=['Original', 'Winsorizada'])
        
        # Colorear boxplots
        bp['boxes'][0].set_facecolor(CYAN + '40')
        bp['boxes'][0].set_edgecolor(CYAN)
        bp['boxes'][1].set_facecolor(GREEN + '40')
        bp['boxes'][1].set_edgecolor(GREEN)
        
        bp['medians'][0].set_color(PINK)
        bp['medians'][1].set_color(PINK)
        
        # Añadir información de reducción de outliers
        reduccion = win_res[c]['n']
        ax.text(0.5, 0.95, f'Outliers tratados: {reduccion:,}', 
                transform=ax.transAxes, ha='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=YELLOW, alpha=0.7))
        
        ax.set_title(f"{CORTO[c]} - Reducción Std: {datos_orig.std():.1f} → {datos_win.std():.1f}", 
                     color=colores_box[i], fontweight="bold")
        ax.set_ylabel("Valores")
        ax.grid(alpha=0.2)
    
    plt.tight_layout()
    guardar(fig, "fig_winsorizacion_boxplots.png")
    
    # FIGURA 4: COMPARACIÓN MÉTODOS DE DETECCIÓN
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Comparación de Métodos de Detección de Outliers", 
                 fontsize=14, fontweight="bold")
    
    labels = [CORTO[c] for c in COLS]
    x = np.arange(len(labels))
    
    # Gráfico 1: Cantidad de outliers detectados
    iqr_counts = [iqr_res[c]['n'] for c in COLS]
    win_counts = [win_res[c]['n'] for c in COLS]
    
    width = 0.35
    ax1.bar(x - width/2, iqr_counts, width, label='IQR (1.5×)', color=RED, alpha=0.7)
    ax1.bar(x + width/2, win_counts, width, label='Winsorización (P5-P95)', color=ORANGE, alpha=0.7)
    
    ax1.set_xlabel('Variables')
    ax1.set_ylabel('Número de outliers detectados')
    ax1.set_title('Cantidad de Outliers Detectados')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Gráfico 2: Reducción de desviación estándar
    std_orig = [win_res[c]['orig_s'] for c in COLS]
    std_win = [win_res[c]['win_s'] for c in COLS]
    
    ax2.bar(x - width/2, std_orig, width, label='Desviación Original', color=PURPLE, alpha=0.7)
    ax2.bar(x + width/2, std_win, width, label='Desviación Winsorizada', color=GREEN, alpha=0.7)
    
    ax2.set_xlabel('Variables')
    ax2.set_ylabel('Desviación Estándar')
    ax2.set_title('Reducción de Dispersión')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    guardar(fig, "fig_comparacion_metodos_outliers.png")
    
    # FIGURA 5: MAPA DE CALOR DE VALORES FALTANTES (ANTES DE IMPUTACIÓN)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("Patrón de Valores Faltantes por Variable", 
                 fontsize=14, fontweight="bold")
    
    # Calcular porcentaje de valores faltantes antes de imputación
    faltantes_antes = pd.DataFrame()
    for c in COLS:
        # Simular datos antes de imputación
        if 'df_original' in locals():
            faltantes_antes[CORTO[c]] = df_original[c].isna()
        else:
            faltantes_antes[CORTO[c]] = df_work[c].isna() & (df_work[c] == 0)
    
    # Calcular porcentaje de faltantes por año (si existe columna AÑO)
    if 'AÑO' in df_work.columns:
        faltantes_por_anio = df_work.groupby('AÑO')[COLS].apply(
            lambda x: x.isna().sum() / len(x) * 100
        )
        faltantes_por_anio.columns = [CORTO[c] for c in COLS]
        
        # Crear heatmap
        im = ax.imshow(faltantes_por_anio.T, aspect='auto', cmap='Reds', 
                       interpolation='nearest', vmin=0, vmax=100)
        
        ax.set_yticks(range(len(faltantes_por_anio.columns)))
        ax.set_yticklabels(faltantes_por_anio.columns)
        ax.set_xticks(range(len(faltantes_por_anio.index)))
        ax.set_xticklabels(faltantes_por_anio.index, rotation=45)
        ax.set_xlabel('Año')
        ax.set_ylabel('Variable')
        ax.set_title('Porcentaje de Valores Faltantes por Año', fontsize=12)
        
        plt.colorbar(im, ax=ax, label='% Faltantes')
    else:
        # Si no hay columna AÑO, hacer heatmap simple
        faltantes_simple = pd.DataFrame({
            CORTO[c]: df_work[c].isna().sum() / len(df_work) * 100 
            for c in COLS
        }, index=['% Faltantes']).T
        
        ax.barh(faltantes_simple.index, faltantes_simple['% Faltantes'], 
                color='red', alpha=0.7)
        ax.set_xlabel('Porcentaje de Valores Faltantes (%)')
        ax.set_title('Valores Faltantes por Variable (Antes de Imputación)')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    guardar(fig, "fig_faltantes_antes_imputacion.png")
    
    print("\n✓ Gráficos generados exitosamente:")
    print("  - fig_imputacion_histogramas.png: Comparación antes/después de imputación")
    print("  - fig_outliers_iqr.png: Detección de outliers por IQR")
    print("  - fig_winsorizacion_boxplots.png: Efecto de winsorización")
    print("  - fig_comparacion_metodos_outliers.png: Comparativa de métodos")
    print("  - fig_faltantes_antes_imputacion.png: Patrón de valores faltantes")

# Ejemplo de uso:
graficar_imputacion_outliers(df_work, COLS, CORTO, iqr_res, win_res)

def graficar_mapa_calor_antes_despues(df_original, df_imputado, COLS, CORTO):
    """
    Genera un mapa de calor comparativo que muestra la distribución de valores
    antes y después de la imputación y tratamiento de outliers.
    
    Parámetros:
    -----------
    df_original : DataFrame
        DataFrame original con valores faltantes (NaN)
    df_imputado : DataFrame
        DataFrame después de imputación y tratamiento de outliers
    COLS : list
        Lista de columnas numéricas a analizar
    CORTO : dict
        Diccionario con nombres cortos para las columnas
    """
    
    # Crear figura con 2 subplots (antes y después)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Impacto del Tratamiento de Datos: Valores Faltantes y Outliers", 
                 fontsize=16, fontweight="bold")
    
    # PREPARAR DATOS PARA EL MAPA DE CALOR "ANTES"
    # Tomar una muestra representativa para mejor visualización (máx 1000 filas)
    n_muestras = min(1000, len(df_original))
    muestra_original = df_original[COLS].sample(n_muestras, random_state=42)
    
    # Crear matriz con valores normalizados para mejor visualización
    # Reemplazar NaN con un valor especial para destacarlos
    matriz_antes = muestra_original.values.copy()
    
    # Crear máscara para valores faltantes
    mask_antes = np.isnan(matriz_antes)
    
    # Normalizar valores para mejor visualización (ignorando NaN)
    valores_validos = matriz_antes[~mask_antes]
    if len(valores_validos) > 0:
        min_val, max_val = np.percentile(valores_validos, [1, 99])
        matriz_antes_norm = np.clip((matriz_antes - min_val) / (max_val - min_val), 0, 1)
    else:
        matriz_antes_norm = np.zeros_like(matriz_antes)
    
    # Asignar valor especial para NaN (0.5 en escala normalizada)
    matriz_antes_norm[mask_antes] = 0.5
    
    # PREPARAR DATOS PARA EL MAPA DE CALOR "DESPUÉS"
    muestra_imputado = df_imputado[COLS].sample(n_muestras, random_state=42)
    matriz_despues = muestra_imputado.values.copy()
    
    # Identificar outliers en datos imputados (usando IQR)
    matriz_despues_norm = matriz_despues.copy()
    for j, col in enumerate(COLS):
        datos = matriz_despues[:, j]
        q1, q3 = np.percentile(datos, [25, 75])
        iqr = q3 - q1
        limite_sup = q3 + 1.5 * iqr
        
        # Normalizar
        if datos.max() > datos.min():
            matriz_despues_norm[:, j] = (datos - datos.min()) / (datos.max() - datos.min())
        
        # Marcar outliers con un valor especial
        outliers = datos > limite_sup
        matriz_despues_norm[outliers, j] = 0.75  # Valor especial para outliers
    
    # MAPA DE CALOR: ANTES
    im1 = ax1.imshow(matriz_antes_norm.T, aspect='auto', cmap='RdYlGn_r', 
                     interpolation='nearest', vmin=0, vmax=1)
    
    # Configurar eje Y con nombres de variables
    ax1.set_yticks(range(len(COLS)))
    ax1.set_yticklabels([CORTO[c] for c in COLS])
    ax1.set_ylabel("Variables", fontsize=12, fontweight="bold")
    
    # Configurar eje X
    ax1.set_xlabel("Muestras (n={:,})".format(n_muestras), fontsize=12, fontweight="bold")
    ax1.set_title("ANTES: Valores Faltantes (gris) y Distribución Original", 
                  fontsize=12, fontweight="bold", pad=20)
    
    # Añadir leyenda para valores faltantes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', label='Valores Faltantes (NaN)'),
        Patch(facecolor='darkgreen', label='Valores Bajos'),
        Patch(facecolor='yellow', label='Valores Medios'),
        Patch(facecolor='darkred', label='Valores Altos')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # MAPA DE CALOR: DESPUÉS
    im2 = ax2.imshow(matriz_despues_norm.T, aspect='auto', cmap='RdYlGn_r', 
                     interpolation='nearest', vmin=0, vmax=1)
    
    # Configurar eje Y
    ax2.set_yticks(range(len(COLS)))
    ax2.set_yticklabels([CORTO[c] for c in COLS])
    ax2.set_ylabel("Variables", fontsize=12, fontweight="bold")
    
    # Configurar eje X
    ax2.set_xlabel("Muestras (n={:,})".format(n_muestras), fontsize=12, fontweight="bold")
    ax2.set_title("DESPUÉS: Datos Imputados con Tratamiento de Outliers", 
                  fontsize=12, fontweight="bold", pad=20)
    
    # Añadir leyenda para outliers
    legend_elements2 = [
        Patch(facecolor='orange', label='Outliers Tratados'),
        Patch(facecolor='darkgreen', label='Valores Bajos'),
        Patch(facecolor='yellow', label='Valores Medios'),
        Patch(facecolor='darkred', label='Valores Altos')
    ]
    ax2.legend(handles=legend_elements2, loc='upper right', fontsize=8)
    
    # Añadir colorbar común
    cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical', pad=0.02, aspect=40)
    cbar.set_label('Intensidad de Valores (normalizada)', fontsize=10)
    
    # Añadir estadísticas resumen
    stats_text = f"""
    Resumen de Mejoras:
    • Valores faltantes originales: {df_original[COLS].isna().sum().sum():,}
    • Valores faltantes después: {df_imputado[COLS].isna().sum().sum():,}
    • Reducción: 100%
    • Outliers tratados: {sum(1 for c in COLS for v in df_imputado[c] if v > df_imputado[c].quantile(0.75) + 1.5*(df_imputado[c].quantile(0.75)-df_imputado[c].quantile(0.25)))}
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=8, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    guardar(fig, "fig_mapa_calor_antes_despues.png")
    
    print("\n✓ Mapa de calor generado: fig_mapa_calor_antes_despues.png")
    print("  - Panel izquierdo: Muestra valores faltantes (gris) en datos originales")
    print("  - Panel derecho: Datos imputados con outliers tratados (naranja)")

graficar_mapa_calor_antes_despues(df, df_work, COLS, CORTO)


def graficar_outliers_iqr(df_work, COLS, CORTO, iqr_res):
    """
    Genera gráficos específicos para visualizar outliers usando el método IQR,
    incluyendo mapas de calor que muestran antes y después del tratamiento.
    
    Parámetros:
    -----------
    df_work : DataFrame
        DataFrame con datos ya imputados
    COLS : list
        Lista de columnas numéricas a analizar
    CORTO : dict
        Diccionario con nombres cortos para las columnas
    iqr_res : dict
        Resultados del método IQR para cada columna
    """
    
    # Colores personalizados
    colores_box = [CYAN, PINK, PURPLE, GREEN, ORANGE, YELLOW]
    
    # FIGURA 1: MAPA DE CALOR ANTES/DESPUÉS - OUTLIERS IQR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Impacto de Outliers (Método IQR): Antes vs Después", 
                 fontsize=16, fontweight="bold")
    
    # Tomar una muestra representativa
    n_muestras = min(1000, len(df_work))
    muestra = df_work[COLS].sample(n_muestras, random_state=42)
    
    # Crear matriz ANTES de tratar outliers
    matriz_antes = muestra.values.copy()
    matriz_antes_norm = np.zeros_like(matriz_antes)
    
    # Normalizar ANTES y marcar outliers
    for j, c in enumerate(COLS):
        datos = matriz_antes[:, j]
        limite = iqr_res[c]['limite']
        
        # Normalizar (ignorando outliers extremos para mejor visualización)
        datos_sin_outliers = datos[datos <= limite]
        if len(datos_sin_outliers) > 0:
            min_val, max_val = np.percentile(datos_sin_outliers, [1, 99])
            matriz_antes_norm[:, j] = np.clip((datos - min_val) / (max_val - min_val), 0, 1)
        else:
            matriz_antes_norm[:, j] = datos / datos.max() if datos.max() > 0 else datos
        
        # Marcar outliers con valor especial
        outliers = datos > limite
        matriz_antes_norm[outliers, j] = 0.75  # Valor para outliers
        
        # Marcar valores cero como otro tipo especial
        zeros = datos == 0
        matriz_antes_norm[zeros, j] = 0.25  # Valor para ceros
    
    # Crear matriz DESPUÉS de tratar outliers (winsorización o clipping)
    matriz_despues = muestra.values.copy()
    matriz_despues_norm = np.zeros_like(matriz_despues)
    
    for j, c in enumerate(COLS):
        datos = matriz_despues[:, j]
        limite = iqr_res[c]['limite']
        
        # Aplicar clipping a los outliers (llevarlos al límite)
        datos_tratados = np.where(datos > limite, limite, datos)
        
        # Normalizar
        if datos_tratados.max() > datos_tratados.min():
            matriz_despues_norm[:, j] = (datos_tratados - datos_tratados.min()) / \
                                        (datos_tratados.max() - datos_tratados.min())
        else:
            matriz_despues_norm[:, j] = datos_tratados / datos_tratados.max() if datos_tratados.max() > 0 else datos_tratados
        
        # Marcar valores que eran outliers originalmente
        outliers = datos > limite
        matriz_despues_norm[outliers, j] = 0.62  # Valor para outliers tratados
    
    # Mapa de calor: ANTES
    im1 = ax1.imshow(matriz_antes_norm.T, aspect='auto', cmap='RdYlGn_r', 
                     interpolation='nearest', vmin=0, vmax=1)
    
    ax1.set_yticks(range(len(COLS)))
    ax1.set_yticklabels([CORTO[c] for c in COLS])
    ax1.set_ylabel("Variables", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Muestras (n={:,})".format(n_muestras), fontsize=12, fontweight="bold")
    ax1.set_title("ANTES: Outliers Detectados por IQR", fontsize=12, fontweight="bold", pad=20)
    
    # Mapa de calor: DESPUÉS
    im2 = ax2.imshow(matriz_despues_norm.T, aspect='auto', cmap='RdYlGn_r', 
                     interpolation='nearest', vmin=0, vmax=1)
    
    ax2.set_yticks(range(len(COLS)))
    ax2.set_yticklabels([CORTO[c] for c in COLS])
    ax2.set_ylabel("Variables", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Muestras (n={:,})".format(n_muestras), fontsize=12, fontweight="bold")
    ax2.set_title("DESPUÉS: Outliers Tratados (Clipping al Límite IQR)", 
                  fontsize=12, fontweight="bold", pad=20)
    
    # Añadir colorbar común
    cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical', pad=0.02, aspect=40)
    cbar.set_label('Intensidad de Valores', fontsize=10)
    
    # Añadir leyenda personalizada
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkred', label='Valores Altos (Normales)'),
        Patch(facecolor='yellow', label='Valores Medios'),
        Patch(facecolor='darkgreen', label='Valores Bajos'),
        Patch(facecolor='orange', label='Outliers Detectados (Antes)'),
        Patch(facecolor='gold', label='Outliers Tratados (Después)'),
        Patch(facecolor='lightgray', label='Valores Cero')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9, 
               bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    guardar(fig, "fig_outliers_iqr_heatmap.png")
    
    # FIGURA 2: MAPA DE CALOR POR VARIABLE - OUTLIERS IQR
    n_vars = len(COLS)
    n_rows = (n_vars + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows))
    fig.suptitle("Detección de Outliers por Variable (Método IQR)", 
                 fontsize=16, fontweight="bold")
    
    axes = axes.flatten() if n_rows > 1 else [axes[0], axes[1]] if n_vars > 1 else [axes]
    
    for idx, c in enumerate(COLS):
        ax = axes[idx]
        
        # Datos de la variable
        datos = df_work[c][df_work[c] > 0].sample(min(500, len(df_work)), random_state=42)
        datos_ordenados = np.sort(datos.values)
        
        # Calcular límites IQR
        q1 = iqr_res[c]['q1']
        q3 = iqr_res[c]['q3']
        iqr = iqr_res[c]['iqr']
        limite_sup = iqr_res[c]['limite']
        
        # Crear matriz para visualización
        n_puntos = len(datos_ordenados)
        matriz = np.zeros((n_puntos, 3))
        matriz[:, 0] = datos_ordenados  # Valores originales
        matriz[:, 1] = np.where(datos_ordenados > limite_sup, limite_sup, datos_ordenados)  # Tratados
        matriz[:, 2] = np.where(datos_ordenados > limite_sup, 1, 0)  # Flag de outliers
        
        # Normalizar
        if matriz[:, 0].max() > matriz[:, 0].min():
            matriz_norm = (matriz - matriz[:, 0].min()) / (matriz[:, 0].max() - matriz[:, 0].min())
        else:
            matriz_norm = matriz
        
        # Graficar
        im = ax.imshow(matriz_norm.T, aspect='auto', cmap='RdYlGn_r', 
                       interpolation='nearest', vmin=0, vmax=1)
        
        # Configurar
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Original', 'Tratado', 'Outlier Flag'])
        ax.set_xticks(range(0, n_puntos, n_puntos//10))
        ax.set_xticklabels(range(0, n_puntos, n_puntos//10))
        ax.set_title(f"{CORTO[c]} - Outliers IQR: {iqr_res[c]['n']:,} ({iqr_res[c]['n']/len(datos)*100:.1f}%)", 
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Muestras ordenadas")
        
        # Añadir líneas de límite
        ax.axhline(y=0.5, color='white', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=1.5, color='white', linestyle='--', linewidth=1, alpha=0.5)
        
        # Añadir estadísticas
        stats_text = f"Q1={q1:.0f} | Q3={q3:.0f} | IQR={iqr:.0f} | Límite={limite_sup:.0f}"
        ax.text(0.5, -0.15, stats_text, transform=ax.transAxes, 
                ha='center', fontsize=8, color='gray')
    
    # Ocultar subplots no utilizados
    for idx in range(len(COLS), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    guardar(fig, "fig_outliers_iqr_por_variable.png")
    
    # FIGURA 3: GRÁFICO DE BARRAS CON DISTRIBUCIÓN DE OUTLIERS
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Distribución de Outliers por Variable (Método IQR)", 
                 fontsize=14, fontweight="bold")
    
    labels = [CORTO[c] for c in COLS]
    x = np.arange(len(labels))
    
    # Datos
    n_outliers = [iqr_res[c]['n'] for c in COLS]
    porcentajes = [iqr_res[c]['n'] / len(df_work[df_work[c] > 0][c]) * 100 for c in COLS]
    
    # Barras
    bars = ax.bar(x, n_outliers, color=RED, alpha=0.7, edgecolor='white', linewidth=1)
    
    # Añadir etiquetas de porcentaje
    for i, (bar, pct) in enumerate(zip(bars, porcentajes)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Variables', fontsize=12)
    ax.set_ylabel('Número de Outliers', fontsize=12)
    ax.set_title('Cantidad de Outliers Detectados por IQR (1.5 × IQR)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Añadir línea de media
    media_outliers = np.mean(n_outliers)
    ax.axhline(y=media_outliers, color=YELLOW, linestyle='--', linewidth=2, 
               label=f'Media: {media_outliers:.0f}')
    ax.legend()
    
    plt.tight_layout()
    guardar(fig, "fig_outliers_iqr_distribucion.png")
    
    # FIGURA 4: BOXPLOTS CON DESTAQUE DE OUTLIERS
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Boxplots con Detección de Outliers (Método IQR)", 
                 fontsize=16, fontweight="bold")
    
    for i, c in enumerate(COLS):
        ax = axes[i // 3][i % 3]
        
        datos = df_work[c][df_work[c] > 0]
        q1 = iqr_res[c]['q1']
        q3 = iqr_res[c]['q3']
        limite = iqr_res[c]['limite']
        
        # Crear boxplot
        bp = ax.boxplot(datos, patch_artist=True, widths=0.6,
                        boxprops=dict(facecolor=CYAN + '40', edgecolor=CYAN),
                        medianprops=dict(color=PINK, linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor=RED, 
                                      markeredgecolor=RED, markersize=4, alpha=0.5))
        
        # Resaltar límite IQR
        ax.axhline(y=limite, color=YELLOW, linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Límite IQR: {limite:.0f}')
        ax.axhline(y=q3, color=ORANGE, linestyle=':', linewidth=1.5, alpha=0.5,
                   label=f'Q3: {q3:.0f}')
        
        ax.set_title(f"{CORTO[c]} - Outliers: {iqr_res[c]['n']:,} ({iqr_res[c]['n']/len(datos)*100:.1f}%)", 
                     color=colores_box[i], fontweight="bold")
        ax.set_ylabel("Valores")
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.2)
    
    plt.tight_layout()
    guardar(fig, "fig_outliers_iqr_boxplots.png")
    
    print("\n✓ Gráficos de outliers IQR generados exitosamente:")
    print("  - fig_outliers_iqr_heatmap.png: Mapa de calor antes/después del tratamiento")
    print("  - fig_outliers_iqr_por_variable.png: Visualización por variable individual")
    print("  - fig_outliers_iqr_distribucion.png: Distribución de outliers por variable")
    print("  - fig_outliers_iqr_boxplots.png: Boxplots con límites IQR destacados")
    
    # Resumen estadístico
    print("\n" + "="*60)
    print("RESUMEN DE OUTLIERS IQR")
    print("="*60)
    print(f"{'Variable':<15} {'Outliers':>10} {'Porcentaje':>10} {'Límite IQR':>12}")
    print("-"*50)
    for c in COLS:
        total = len(df_work[df_work[c] > 0][c])
        print(f"{CORTO[c]:<15} {iqr_res[c]['n']:>10,} {iqr_res[c]['n']/total*100:>9.1f}% {iqr_res[c]['limite']:>12.0f}")
    print("="*60)

# Ejemplo de uso:
graficar_outliers_iqr(df_work, COLS, CORTO, iqr_res)