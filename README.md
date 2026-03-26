# 📊 Análisis de Datos — Universidades de Bolivia (2001–2016)

Análisis estadístico del dataset de universidades bolivianas que incluye detección de outliers, tratamiento de datos faltantes, métricas descriptivas y visualizaciones.

## 📁 Estructura del Proyecto

```
PROYECTO/
├── Codigo/
│   ├── analisis.py                      # Script principal
│   └── universities_bolivia_dataset.csv # Dataset
├── informe/
│   ├── fig1_boxplots.png                # Diagramas de caja
│   ├── fig2_dispersion.png              # Nube de dispersión
│   ├── fig3_barras_area.png             # Barras por área
│   ├── fig4_barras_ciudad.png           # Barras por departamento
│   ├── fig5_evolucion.png               # Evolución temporal
│   ├── fig6_comparativa_outliers.png    # IQR vs Winsorización
│   ├── fig7_media_vs_mediana.png        # Media vs Mediana
│   └── fig8_publica_vs_privada.png      # Pública vs Privada
├── Presentacion/
│   └── (archivos de presentación)
├── requirements.txt
└── README.md
```

## 🔍 ¿Qué hace el análisis?

| Sección | Descripción |
|---------|-------------|
| **Datos faltantes** | Identifica 2,301 filas sin datos (22.4%) y las imputa con mediana por grupo |
| **Outliers IQR** | Detecta outliers con regla Q1−1.5×IQR / Q3+1.5×IQR (~9–13%) |
| **Winsorización** | Recorta valores extremos en percentiles 5%–95%, reduce Std 33–40% |
| **Métricas** | Media, mediana, desviación estándar, asimetría, curtosis por variable |
| **Gráficos** | 8 figuras: box plots, dispersión, barras, líneas temporales |

## 📈 Dataset

- **Fuente**: Universidades de Bolivia
- **Registros**: 10,263 filas × 11 columnas
- **Período**: 2001 – 2016
- **Variables**: Matriculados, Nuevos inscritos y Titulados (hombres/mujeres)
- **Categorías**: 9 departamentos, 10 áreas de conocimiento, educación pública/privada

## 🚀 Instalación y Ejecución

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/PROYECTO.git
cd PROYECTO

# 2. Instalar dependencias
# en sta parte deben estar dentro de la carpeto de codigo
py -m venv .venv

#luego deben de activarlo
.venv\Scripts\activate

#por ultimo installar los requerimientos
pip install -r requirements.txt

# 3. Ejecutar el análisis
cd Codigo
python analisis.py
```

Los 8 gráficos se guardan automáticamente en la carpeta `informe/`.

## 🛠️ Tecnologías

- **Python 3.8+**
- **pandas** — manipulación de datos
- **numpy** — cálculos numéricos
- **scipy** — winsorización y estadística
- **matplotlib** — visualizaciones

## 📊 Hallazgos Principales

1. **Cruce de género (2012)**: Las mujeres superaron a los hombres en matrícula total
2. **Asimetría alta**: Pocas carreras concentran la mayoría de estudiantes
3. **Brecha por tipo**: Universidades públicas tienen ~2.5× más matriculados que privadas
4. **La Paz y Santa Cruz** lideran en promedio de matriculados por departamento
