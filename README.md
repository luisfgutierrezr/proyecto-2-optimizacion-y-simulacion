# Análisis de Colas TransMilenio - Modelo M/M/1

Este proyecto implementa un análisis completo de teoría de colas aplicado a una taquilla del sistema TransMilenio, utilizando el modelo M/M/1 y simulación con SimPy.

## 📋 Descripción del Proyecto

El proyecto analiza datos reales de observación de una taquilla de TransMilenio en dos franjas horarias diferentes:
- **9 AM**: 43 clientes observados durante 10 minutos
- **4 PM**: 31 clientes observados durante 10 minutos

Se utiliza la teoría de colas M/M/1 para modelar el sistema y proponer mejoras basadas en el análisis de rendimiento.

## 🗂️ Estructura del Proyecto

```
proyecto-2/
├── datos_9am.csv                          # Datos de observación 9 AM
├── datos_4pm.csv                          # Datos de observación 4 PM
├── analisis_colas_transmilenio.py         # Script principal de análisis
├── requirements.txt                       # Dependencias del proyecto
└── README.md                             # Este archivo
```

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### Dependencias Principales
- `simpy==4.0.1` - Simulación de sistemas discretos
- `pandas==2.1.4` - Manipulación y análisis de datos
- `numpy==1.24.3` - Computación numérica
- `matplotlib==3.8.2` - Visualización de datos
- `scipy==1.11.4` - Funciones estadísticas y científicas
- `seaborn==0.13.0` - Visualización estadística avanzada

## 🎯 Uso del Proyecto

### Ejecución del Análisis

```bash
python analisis_colas_transmilenio.py
```

### Salida del Programa

El script generará:

1. **Reporte en consola** con métricas detalladas:
   - Estadísticas descriptivas de tiempos de servicio
   - Tasas de llegada (λ) y servicio (μ)
   - Factor de utilización (ρ)
   - Métricas teóricas M/M/1 (L, Lq, W, Wq)

2. **Gráficas guardadas**:
   - `histogramas_tiempos_servicio.png` - Distribución de tiempos de servicio
   - `comparacion_metricas.png` - Comparación de métricas entre franjas
   - `qq_plots_exponencial.png` - Verificación de distribución exponencial
   - `analisis_eficiencia.png` - Análisis de eficiencia del sistema

3. **Propuesta de mejora** basada en el análisis de utilización

## 📊 Métricas Analizadas

### Métricas Teóricas M/M/1

- **L**: Número promedio de clientes en el sistema
- **Lq**: Número promedio de clientes en cola
- **W**: Tiempo promedio en el sistema
- **Wq**: Tiempo promedio en cola
- **ρ**: Factor de utilización (λ/μ)

### Análisis Estadístico

- Verificación de distribución exponencial mediante Q-Q plots
- Prueba de Kolmogorov-Smirnov para bondad de ajuste
- Comparación de eficiencia entre franjas horarias

## 🔍 Interpretación de Resultados

### Factor de Utilización (ρ)
- **ρ < 0.7**: Sistema eficiente con buena capacidad ociosa
- **0.7 ≤ ρ < 0.8**: Sistema con alta utilización, monitorear
- **ρ ≥ 0.8**: Sistema cerca de saturación, considerar mejoras

### Recomendaciones Automáticas
El sistema genera recomendaciones automáticas basadas en:
- Nivel de utilización del sistema
- Comparación entre franjas horarias
- Análisis de eficiencia operativa

## 📈 Ejemplo de Salida

```
ANÁLISIS DE COLAS TRANSMILENIO - MODELO M/M/1
============================================================

Cargando datos de observación...
Datos 9 AM: 43 observaciones
Datos 4 PM: 31 observaciones

📊 FRANJA HORARIA: 9 AM
------------------------------
Número de clientes observados: 43
Tiempo promedio de servicio: 13.45 segundos
Tasa de llegada (λ): 0.0717 clientes/seg
Tasa de servicio (μ): 0.0744 clientes/seg
Factor de utilización (ρ): 0.964
Clientes promedio en sistema (L): 26.694
Tiempo promedio en cola (Wq): 359.67 segundos

🚨 ALERTA: El sistema de 4 PM está cerca de la saturación (ρ > 0.8)
RECOMENDACIÓN: Implementar una segunda taquilla durante la hora pico vespertina
```

## 🛠️ Personalización

### Modificar Parámetros de Simulación

Puedes modificar los parámetros en la clase `AnalisisColasTransmilenio`:

```python
# Cambiar tiempo de simulación
tiempo_simulacion = 600  # segundos (10 minutos)

# Modificar umbrales de alerta
if rho > 0.8:  # Umbral de saturación
    print("Sistema cerca de saturación")
```

### Agregar Nuevos Datos

1. Crear nuevos archivos CSV con formato:
   ```csv
   persona,tiempo_segundos
   1,tiempo1
   2,tiempo2
   ...
   ```

2. Modificar el método `cargar_datos()` para incluir nuevos archivos

## 📚 Referencias Teóricas

- **Modelo M/M/1**: Sistema de colas con llegadas Poisson, servicio exponencial y un servidor
- **Teoría de Colas**: Aplicación de procesos estocásticos a sistemas de servicio
- **SimPy**: Framework de simulación de eventos discretos en Python

## 👥 Autor

Proyecto desarrollado para el curso de Optimización y Simulación - Pontificia Universidad Javeriana

## 📝 Licencia

Este proyecto es para fines académicos y educativos.

---

**Nota**: Asegúrate de tener todas las dependencias instaladas antes de ejecutar el análisis. El programa generará gráficas que se guardarán automáticamente en el directorio actual.
