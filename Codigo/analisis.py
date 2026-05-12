# %%
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# %%
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


# %%
def guardar(fig, nombre):
    """Guarda figura en la carpeta informe."""
    ruta = os.path.join(OUT, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {nombre}")


# %%
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


# %%
# ================================================================
# 2. DATOS FALTANTES
# ================================================================
# El dataset no tiene NaN literales, pero las filas donde TODAS
# las columnas numéricas valen 0 representan registros sin datos
# reportados (carrera sin actividad ese año).  Se detectan,
# se convierten a NaN y se imputan con la mediana por grupo
# (ÁREA × TIPO DE EDUCACIÓN) para respetar el contexto de cada
# carrera en vez de usar una mediana global.
# ================================================================
print("\n" + "=" * 60)
print("2. DATOS FALTANTES")
print("=" * 60)

# %%
# ── 2a. Diagnóstico: NaN literales ──────────────────────────────
nan_lit = df[COLS].isna().sum()
print("\n  NaN literales por columna:")
for c in COLS:
    print(f"    {CORTO[c]:<8} {nan_lit[c]:>6,}")
print(f"  Total NaN literales: {nan_lit.sum()}")

# ── 2b. Diagnóstico: filas «todo-cero» ──────────────────────────
# Un registro con todos los valores en 0 no aporta información;
# tratarlos como NaN permite imputarlos correctamente.
todo_cero = df[COLS].sum(axis=1) == 0
print(f"\n  Filas sin datos (todos ceros): {todo_cero.sum():,}  "
      f"({todo_cero.mean()*100:.1f}% del total)")
print(f"  Filas con al menos un valor  : {(~todo_cero).sum():,}")

df_work = df.copy()
for c in COLS:
    df_work.loc[todo_cero, c] = np.nan

# %%

    # ── 2c. Tabla resumen antes de imputar ──────────────────────────
# La diferencia entre media y mediana revela distribuciones muy
# asimétricas (skew > 3 en todas las variables), lo que justifica
# usar la mediana como imputador en lugar de la media.
print(f"\n  {'Variable':<12} {'Faltantes':>10} {'% total':>8} "
      f"{'Media':>9} {'Mediana':>9} {'Diferencia%':>12}")
print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*9} {'-'*9} {'-'*12}")
for c in COLS:
    na  = df_work[c].isna().sum()
    pct = na / len(df_work) * 100
    me  = df_work[c].mean()
    md  = df_work[c].median()
    dif = (me - md) / md * 100 if md != 0 else 0
    print(f"  {CORTO[c]:<12} {na:>10,} {pct:>7.1f}% "
          f"{me:>9.1f} {md:>9.1f} {dif:>+11.1f}%")

# ── 2d. Imputación con mediana por grupo ────────────────────────
# Estrategia de dos pasos:
#   1) Mediana del grupo ÁREA × TIPO DE EDUCACIÓN  →  respeta
#      el perfil típico de cada área y tipo de universidad.
#   2) Mediana global de la columna  →  fallback para grupos
#      que también quedaron sin datos.
for c in COLS:
    mediana_grupo = df_work.groupby(
        ["ÁREA", "TIPO DE EDUCACIÓN"])[c].transform("median")
    df_work[c] = df_work[c].fillna(mediana_grupo).fillna(df_work[c].median())

print(f"\n  Método : mediana por grupo (ÁREA + TIPO DE EDUCACIÓN)")
print(f"  NaN restantes tras imputar: {df_work[COLS].isna().sum().sum()}")

# %%
# ── 2e. Gráfico: patrón de faltantes + media vs mediana ─────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("2. Datos Faltantes", fontsize=15, fontweight="bold")

# Izquierda — barras de faltantes por variable
ax = axes[0]
colores_na = [CYAN, PINK, PURPLE, GREEN, ORANGE, YELLOW]
na_counts  = [df[COLS].isna().sum()[c] + todo_cero.sum() for c in COLS]
# (NaN literales + filas todo-cero convertidas)
na_real = [df_work.isna().sum()[c] + 0 for c in COLS]   # después de marcar
bars = ax.bar([CORTO[c] for c in COLS], [todo_cero.sum()] * len(COLS),
              color=colores_na, alpha=0.0)               # placeholder altura
# recalcular con la copia marcada
na_vals = []
df_tmp = df.copy()
for c in COLS:
    df_tmp.loc[todo_cero, c] = np.nan
    na_vals.append(df_tmp[c].isna().sum())
ax.cla()
bars = ax.bar([CORTO[c] for c in COLS], na_vals,
              color=colores_na, edgecolor="white", linewidth=0.6)
for b, v in zip(bars, na_vals):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
            f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_title("Valores faltantes por variable\n(filas todo-cero → NaN)", fontsize=11)
ax.set_ylabel("Registros sin dato")
ax.grid(axis="y", alpha=0.25)

# Derecha — media vs mediana (justificación del imputador)
ax2 = axes[1]
x  = np.arange(len(COLS))
medias   = [df_work[c].mean()   for c in COLS]
medianas = [df_work[c].median() for c in COLS]
ax2.bar(x - 0.18, medias,   0.35, label="Media",   color=PINK,  alpha=0.85)
ax2.bar(x + 0.18, medianas, 0.35, label="Mediana", color=CYAN, alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels([CORTO[c] for c in COLS])
ax2.set_title("Media vs Mediana\n(asimetría justifica usar mediana)", fontsize=11)
ax2.set_ylabel("Promedio de estudiantes")
ax2.legend()
ax2.grid(axis="y", alpha=0.25)

plt.tight_layout()
guardar(fig, "fig_datos_faltantes.png")

# %%
# ================================================================
# 3. DETECCIÓN Y TRATAMIENTO DE OUTLIERS
# ================================================================
# IMPORTANTE — distinción conceptual:
#
#   DETECCIÓN  → identificar qué valores son atípicos.
#                Herramienta usada: regla IQR (Q3 + 1.5 × IQR).
#
#   TRATAMIENTO → decidir qué hacer con los outliers detectados.
#                 Se comparan dos estrategias:
#                   A) Eliminación  — borrar las filas con outliers.
#                   B) Winsorización — reemplazar los extremos por
#                      los valores de los percentiles P5 y P95,
#                      conservando todas las filas.
#
# Comparar IQR con Winsorización como si fueran del mismo tipo
# es un error: el IQR es una regla de DETECCIÓN, mientras que
# eliminación y winsorización son estrategias de TRATAMIENTO.
# La comparación correcta es: Eliminación vs. Winsorización.
# ================================================================
print("\n" + "=" * 60)
print("3. DETECCIÓN Y TRATAMIENTO DE OUTLIERS")
print("=" * 60)

# ── 3a. DETECCIÓN: Regla IQR ────────────────────────────────────
# Se identifican como outliers los valores que superan Q3 + 1.5×IQR
# (extremo superior).  Solo se consideran registros con valor > 0.
print("\n  [DETECCIÓN] Regla IQR — Q3 + 1.5 × IQR")
print(f"  {'Var':<9} {'Q1':>7} {'Q3':>7} {'IQR':>7} "
      f"{'Lím.sup':>9} {'Outliers':>9} {'%':>6}")
print(f"  {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*9} {'-'*6}")

iqr_res = {}
for c in COLS:
    v      = df_work[c][df_work[c] > 0]
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr    = q3 - q1
    limite = q3 + 1.5 * iqr
    n_out  = (v > limite).sum()
    iqr_res[c] = {"q1": q1, "q3": q3, "iqr": iqr, "limite": limite, "n": n_out}
    print(f"  {CORTO[c]:<9} {q1:>7.0f} {q3:>7.0f} {iqr:>7.0f} "
          f"{limite:>9.0f} {n_out:>9,} {n_out/len(v)*100:>5.1f}%")

# ── 3b. TRATAMIENTO A: Eliminación ──────────────────────────────
# Se eliminan las filas donde AL MENOS UNA columna supera su
# límite IQR.  Ventaja: elimina el ruido.
# Desventaja: se pierde información — reducción del dataset.
mask_outlier = pd.Series(False, index=df_work.index)
for c in COLS:
    mask_outlier |= (df_work[c] > iqr_res[c]["limite"])

df_elim = df_work[~mask_outlier].copy()
filas_elim = mask_outlier.sum()

print(f"\n  [TRATAMIENTO A] Eliminación de filas con outlier IQR")
print(f"  Filas eliminadas : {filas_elim:,} ({filas_elim/len(df_work)*100:.1f}%)")
print(f"  Filas restantes  : {len(df_elim):,}")
print(f"\n  {'Var':<9} {'Media orig':>11} {'→ Elim':>9} "
      f"{'Std orig':>10} {'→ Elim':>9} {'Red.Std%':>9}")
print(f"  {'-'*9} {'-'*11} {'-'*9} {'-'*10} {'-'*9} {'-'*9}")
elim_res = {}
for c in COLS:
    orig_m = df_work[c][df_work[c] > 0].mean()
    orig_s = df_work[c][df_work[c] > 0].std()
    new_m  = df_elim[c][df_elim[c] > 0].mean()
    new_s  = df_elim[c][df_elim[c] > 0].std()
    red    = (orig_s - new_s) / orig_s * 100
    elim_res[c] = {"orig_m": orig_m, "new_m": new_m,
                   "orig_s": orig_s, "new_s": new_s, "red": red}
    print(f"  {CORTO[c]:<9} {orig_m:>11.1f} {new_m:>9.1f} "
          f"{orig_s:>10.1f} {new_s:>9.1f} {red:>8.1f}%")

# ── 3c. TRATAMIENTO B: Winsorización (P5–P95) ───────────────────
# Los valores extremos se recortan al P5 (inferior) y al P95
# (superior), reemplazándolos por esos percentiles.
# Ventaja: no se pierde ninguna fila.
# Desventaja: modifica los valores originales.
print(f"\n  [TRATAMIENTO B] Winsorización P5–P95")
print(f"  {'Var':<9} {'Media orig':>11} {'→ Win':>9} "
      f"{'Std orig':>10} {'→ Win':>9} {'Afect.':>8} {'Red.Std%':>9}")
print(f"  {'-'*9} {'-'*11} {'-'*9} {'-'*10} {'-'*9} {'-'*8} {'-'*9}")

win_res = {}
for c in COLS:
    v    = df_work[c][df_work[c] > 0]
    w    = stats.mstats.winsorize(v, limits=[0.05, 0.05])
    n_af = int(((v < np.percentile(v, 5)) | (v > np.percentile(v, 95))).sum())
    red  = (v.std() - float(w.std())) / v.std() * 100
    win_res[c] = {"orig_m": v.mean(), "win_m": float(w.mean()),
                  "orig_s": v.std(),  "win_s": float(w.std()),
                  "n": n_af, "red": red}
    print(f"  {CORTO[c]:<9} {v.mean():>11.1f} {float(w.mean()):>9.1f} "
          f"{v.std():>10.1f} {float(w.std()):>9.1f} {n_af:>8,} {red:>8.1f}%")

# ── 3d. COMPARATIVA: Eliminación vs. Winsorización ──────────────
# Ahora sí comparamos dos estrategias de TRATAMIENTO entre sí.
print(f"\n  [COMPARATIVA] Eliminación vs. Winsorización")
print(f"  (ambas usan IQR como criterio de detección)")
print(f"  {'Var':<9} {'Red.Std Elim%':>14} {'Red.Std Win%':>13}  "
      f"{'Filas perdidas':>15}")
print(f"  {'-'*9} {'-'*14} {'-'*13}  {'-'*15}")
for c in COLS:
    print(f"  {CORTO[c]:<9} {elim_res[c]['red']:>13.1f}% "
          f"{win_res[c]['red']:>12.1f}%  "
          f"{filas_elim:>12,} vs 0")
print(f"\n  → Eliminación reduce más la Std pero descarta "
      f"{filas_elim:,} filas ({filas_elim/len(df_work)*100:.1f}%).")
print(f"  → Winsorización conserva todas las filas a cambio de")
print(f"    modificar los valores extremos.")
print(f"  → En este dataset se prefiere Winsorización para no")
print(f"    perder información de carreras con matrículas altas.")

# ── 3e. Gráfico: Box plots (detección IQR) ──────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("3a. Detección de Outliers con Regla IQR",
             fontsize=16, fontweight="bold")
colores_box = [CYAN, PINK, PURPLE, GREEN, ORANGE, YELLOW]
for i, c in enumerate(COLS):
    ax   = axes[i // 3][i % 3]
    data = df_work[c][df_work[c] > 0]
    ax.boxplot(
        data, patch_artist=True, widths=0.5,
        boxprops    =dict(facecolor=colores_box[i] + "40",
                          edgecolor=colores_box[i], linewidth=1.5),
        medianprops =dict(color=PINK, linewidth=2.5),
        whiskerprops=dict(color=colores_box[i], linewidth=1.2),
        capprops    =dict(color=colores_box[i], linewidth=1.5),
        flierprops  =dict(marker=".", color=RED, markersize=2, alpha=0.3),
    )
    ax.set_title(CORTO[c], color=colores_box[i], fontweight="bold")
    ax.set_xlabel(
        f"Outliers detectados: {iqr_res[c]['n']:,}  |  "
        f"Límite sup.: {iqr_res[c]['limite']:.0f}",
        fontsize=8, color=RED,
    )
    ax.set_xticks([])
    ax.set_ylabel("Cantidad de estudiantes")
    ax.grid(alpha=0.2)
plt.tight_layout()
guardar(fig, "fig_outliers_deteccion_iqr.png")

# ── 3f. Gráfico: Eliminación vs. Winsorización ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "3b. Tratamiento de Outliers — Eliminación vs. Winsorización\n"
    "(detección previa con regla IQR en ambos casos)",
    fontsize=13, fontweight="bold",
)
labels = [CORTO[c] for c in COLS]
x = np.arange(len(labels))

# Panel izquierdo — reducción de Std
ax = axes[0]
ax.bar(x - 0.18, [elim_res[c]["red"] for c in COLS], 0.35,
       label="Eliminación", color=RED,   alpha=0.85)
ax.bar(x + 0.18, [win_res[c]["red"]  for c in COLS], 0.35,
       label="Winsorización", color=GREEN, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
ax.set_title("Reducción de Desv. Estándar (%)", fontsize=11)
ax.set_ylabel("Reducción (%)")
ax.legend(); ax.grid(axis="y", alpha=0.2)

# Panel central — media antes/después por tratamiento
ax2 = axes[1]
medias_orig = [df_work[c][df_work[c] > 0].mean() for c in COLS]
medias_elim = [elim_res[c]["new_m"] for c in COLS]
medias_win  = [win_res[c]["win_m"]  for c in COLS]
ax2.bar(x - 0.25, medias_orig, 0.24, label="Original",     color=BLUE,   alpha=0.85)
ax2.bar(x,        medias_elim, 0.24, label="Eliminación",  color=RED,    alpha=0.85)
ax2.bar(x + 0.25, medias_win,  0.24, label="Winsorización",color=GREEN,  alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15)
ax2.set_title("Media original vs. tras tratamiento", fontsize=11)
ax2.set_ylabel("Media de estudiantes")
ax2.legend(); ax2.grid(axis="y", alpha=0.2)

# Panel derecho — costo: filas perdidas vs. filas modificadas
ax3 = axes[2]
filas_mod_win = [win_res[c]["n"] for c in COLS]
ax3.bar(x - 0.18, [filas_elim] * len(COLS), 0.35,
        label=f"Filas eliminadas ({filas_elim:,})", color=RED,    alpha=0.85)
ax3.bar(x + 0.18, filas_mod_win,              0.35,
        label="Filas modificadas (win)",           color=ORANGE, alpha=0.85)
ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=15)
ax3.set_title("Costo de cada tratamiento\n(filas perdidas vs. modificadas)", fontsize=11)
ax3.set_ylabel("Cantidad de filas")
ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.2)

plt.tight_layout()
guardar(fig, "fig_outliers_tratamiento_comparativa.png")


# %%
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
# %%
# ================================================================
# 5. NORMALIZACIÓN Y ENTANDARIZACIÓN
# ================================================================

print("""
- Normalización: Lleva los valores a 0 y 1. Se recomienda para redes neuronales.

- Estandarizacion: Lleva los valores con media 0 y desviación estandar
de 1. Se recomienda para SVM, arboles de desición, PCA. Esta no se aplica al
tiempo por que se pierde la relación temporal

Aplicaremos ambos casos por si acaso para en un futuro usar en diferentes casos.
Usaremos Scikit-Learn, ya que nos permitirá poder realizar el train spit posterior.
""")

# %%
print("\n" + "=" * 60)
print("VARIABLES ESTANDARIZADAS")
print("=" * 60)

COLS_TO_STANDARDIZE = COLS.copy()
print("Columnas a estandarizar:", COLS_TO_STANDARDIZE)
scaler_standard_scaler = StandardScaler()
df_work_standard = df_work.copy()
df_work_standard[COLS_TO_STANDARDIZE] = scaler_standard_scaler.fit_transform(df_work_standard[COLS])

print("\nMuestra:\n")
print(df_work_standard[COLS_TO_STANDARDIZE].sample(n=5))
print("\nDescripción:\n")
print(df_work_standard[COLS_TO_STANDARDIZE].describe())

# %%
print("\n" + "=" * 60)
print("VARIABLES NORMALIZADAS")
print("=" * 60)

COLS_TO_NORMALIZE = COLS.copy()
print("Columnas a estandarizar:", COLS_TO_NORMALIZE)
scaler_min_max_scaler = MinMaxScaler()
df_work_min_max_scaler = df_work.copy()
df_work_min_max_scaler[COLS_TO_NORMALIZE] = scaler_min_max_scaler.fit_transform(df_work_min_max_scaler[COLS])

print("\nMuestra:\n")
print(df_work_min_max_scaler[COLS_TO_NORMALIZE].sample(n=5))
print("\nDescripción:\n")
print(df_work_min_max_scaler[COLS_TO_NORMALIZE].describe())

# %%

print("""
En función de los modelos escogidos posteriormente se podrá decidir por
una de las dos transformaciones.
""")

# %%
# ================================================================
# 6. CONVERSIÓN DE TIPOS DE DATOS
# ================================================================

print("""
Para ello vemos que las columnas que se pueden cambiar de tipo son:
  'ÁREA', 'CARRERA', 'CIUDAD', 'AÑO', 'TIPO DE EDUCACIÓN'
""")

print("Viendo de que tipo son las columnas:")
print(df_work[["ÁREA", "CARRERA", "CIUDAD", "AÑO", "TIPO DE EDUCACIÓN"]].dtypes)

print("Creando copia de seguridad")
df_change_type = df_work.copy()

# %%
CATEGORICAL_COLS = ["ÁREA", "CARRERA", "CIUDAD", "TIPO DE EDUCACIÓN"]
print(f"Convirtiendo {CATEGORICAL_COLS} a categórica")
df_change_type[CATEGORICAL_COLS] = df_change_type[CATEGORICAL_COLS].astype("category")

# %%
print("""
No es posible convertir el año a pd.date_time por que no se tiene ni meses ni dias.
Los caminos que se recomiendas depende del enfoque que le daremos:
    - Serie de tiempo: Se puede estandarizar
    - Sin enfoque temporal: Se puede tratar como una variable categórica
""")
# TIME_COLS = ["AÑO"]
# print(f"Convirtiendo {TIME_COLS} a tiempo")
# df_change_type[TIME_COLS] = pd.to_datetime(df_change_type[TIME_COLS], format="%Y")
df_change_type["AÑO"] = df_change_type["AÑO"].astype("category")

# %%
print("Viendo de que tipo son las columnas:")
print(df_change_type[["ÁREA", "CARRERA", "CIUDAD", "AÑO", "TIPO DE EDUCACIÓN"]].dtypes)


# %%
# ================================================================
# 5. GRÁFICOS
# ================================================================
print("\n" + "=" * 60)
print("5. GENERANDO GRÁFICOS")
print("=" * 60)

colores_box = [CYAN, PINK, PURPLE, GREEN, ORANGE, YELLOW]

# %%
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

# %%
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

# %%
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

# %%
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

# %%
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

# %%
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

# %%
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

# %%
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
