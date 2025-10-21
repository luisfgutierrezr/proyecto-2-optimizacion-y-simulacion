# Análisis de Colas TransMilenio - Modelo M/M/1

## Descripción del Proyecto

Este proyecto implementa un análisis completo de teoría de colas para evaluar el rendimiento de una taquilla de TransMilenio utilizando el modelo M/M/1. El análisis se realiza en dos franjas horarias diferentes (9 AM y 4 PM) para identificar patrones de demanda y proponer mejoras operativas.

## Modelo M/M/1

El modelo M/M/1 es un sistema de colas con las siguientes características:

- **M (Llegadas)**: Proceso de llegada de Poisson con tasa λ
- **M (Servicio)**: Tiempos de servicio con distribución exponencial con tasa μ  
- **1**: Un solo servidor

### Condición de Estabilidad

Para que el sistema sea estable, se requiere que ρ = λ/μ < 1, donde:
- λ: Tasa de llegada de clientes
- μ: Tasa de servicio del servidor
- ρ: Factor de utilización del servidor

## Estructura del Proyecto

```
proyecto-2-optimizacion-y-simulacion/
├── analisis_colas_transmilenio.py    # Script principal de análisis
├── datos_9am.csv                    # Datos de observación 9 AM
├── datos_4pm.csv                     # Datos de observación 4 PM
├── requirements.txt                   # Dependencias del proyecto
├── histogramas_tiempos_servicio.png   # Gráfico de distribuciones
├── comparacion_metricas.png           # Comparación entre franjas
├── qq_plots_exponencial.png          # Verificación de distribución
├── analisis_eficiencia.png           # Análisis de eficiencia
└── README.md                         # Este archivo
```

## Requisitos del Sistema

### Dependencias Python

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
simpy>=4.0.0
```

### Instalación

```bash
pip install -r requirements.txt
```

## Formato de Datos

Los archivos CSV deben contener al menos una columna `tiempo_segundos` con los tiempos de servicio observados para cada cliente. Ejemplo:

```csv
tiempo_segundos
15.2
23.1
18.7
...
```

## Uso del Programa

### Ejecución Básica

```bash
python analisis_colas_transmilenio.py
```

### Funcionalidades Principales

1. **Carga de Datos**: Lee archivos CSV con tiempos de servicio
2. **Cálculo de Métricas**: Aplica fórmulas del modelo M/M/1
3. **Verificación Estadística**: Prueba de bondad de ajuste para distribución exponencial
4. **Visualización**: Genera gráficos de análisis
5. **Recomendaciones**: Propone mejoras basadas en los resultados

## Métricas Calculadas

### Parámetros del Sistema

- **λ (lambda)**: Tasa de llegada de clientes (clientes/segundo)
- **μ (mu)**: Tasa de servicio del servidor (clientes/segundo)
- **ρ (rho)**: Factor de utilización del servidor

### Métricas de Rendimiento

- **L**: Número promedio de clientes en el sistema
- **Lq**: Número promedio de clientes en la cola
- **W**: Tiempo promedio en el sistema
- **Wq**: Tiempo promedio en la cola

### Fórmulas del Modelo M/M/1

```
L = ρ / (1 - ρ)
Lq = ρ² / (1 - ρ)
W = 1 / (μ - λ)
Wq = ρ / (μ - λ)
```

## Visualizaciones Generadas

### 1. Histogramas de Tiempos de Servicio
- Muestra la distribución empírica de los tiempos de servicio
- Permite identificar patrones en los datos observados

### 2. Comparación de Métricas
- Gráficos de barras comparando métricas entre franjas horarias
- Facilita la identificación de diferencias en el rendimiento

### 3. Q-Q Plots
- Verifica si los datos siguen distribución exponencial
- Crucial para validar las asunciones del modelo M/M/1

### 4. Análisis de Eficiencia
- Muestra la capacidad ociosa del sistema como porcentaje
- Código de colores: Verde (>20%), Naranja (10-20%), Rojo (<10%)

## Interpretación de Resultados

### Factor de Utilización (ρ)

- **ρ < 0.7**: Sistema cómodo, baja probabilidad de colas largas
- **0.7 ≤ ρ < 0.8**: Sistema moderadamente cargado
- **0.8 ≤ ρ < 0.9**: Sistema altamente cargado, colas frecuentes
- **ρ ≥ 0.9**: Sistema crítico, colas muy largas
- **ρ ≥ 1.0**: Sistema inestable, colas crecen indefinidamente

### Recomendaciones por Nivel de Utilización

#### ρ > 0.8 (Sistema Crítico)
- Implementar servidor adicional
- Considerar automatización
- Reorganizar horarios de personal

#### 0.7 ≤ ρ < 0.8 (Sistema Cargado)
- Monitorear constantemente
- Preparar personal adicional para picos
- Implementar sistema de información al cliente

#### ρ < 0.7 (Sistema Cómodo)
- Mantener operación actual
- Considerar optimizaciones menores

## Análisis Estadístico

### Prueba de Kolmogorov-Smirnov

Se utiliza para verificar si los tiempos de servicio siguen distribución exponencial:

- **H0**: Los datos siguen distribución exponencial
- **H1**: Los datos NO siguen distribución exponencial
- **Criterio**: Si p-valor > 0.05, no se rechaza H0

### Q-Q Plots

Los gráficos Q-Q muestran la proximidad de los datos a la distribución teórica:
- Puntos alineados con la línea diagonal: buena bondad de ajuste
- Desviaciones significativas: posible violación de asunciones

## Simulación con SimPy

El código incluye una implementación de simulación discreta usando SimPy para validar los resultados teóricos. La simulación modela:

- Llegadas de clientes con distribución exponencial
- Tiempos de servicio con distribución exponencial
- Cola FIFO (First In, First Out)
- Un solo servidor

## Limitaciones del Modelo

### Asunciones del Modelo M/M/1

1. **Llegadas de Poisson**: Los clientes llegan independientemente
2. **Servicio Exponencial**: Los tiempos de servicio son aleatorios
3. **Un Solo Servidor**: Capacidad limitada a un servidor
4. **Cola Infinita**: No hay límite en el número de clientes
5. **Disciplina FIFO**: Primer cliente en llegar, primero en ser atendido

### Cuándo NO Usar M/M/1

- Servicios con múltiples fases
- Sistemas con prioridades
- Colas con capacidad limitada
- Servicios no exponenciales (determinísticos, uniformes, etc.)

## Mejoras Propuestas

### Implementación de Segunda Taquilla

Para sistemas con ρ > 0.8, se propone:

1. **Análisis M/M/2**: Modelo con dos servidores
2. **Redistribución de Carga**: Balancear demanda entre servidores
3. **Horarios Flexibles**: Personal adicional en horas pico

### Automatización Parcial

- **Transacciones Simples**: Recarga de tarjetas automática
- **Información Digital**: Pantallas con tiempos de espera
- **Reservas Online**: Reducir demanda en taquilla física

## Casos de Uso

### Gestión Operativa
- Planificación de personal
- Optimización de horarios
- Análisis de capacidad

### Toma de Decisiones
- Inversión en infraestructura
- Implementación de tecnología
- Reorganización de procesos

### Monitoreo Continuo
- Indicadores de rendimiento
- Alertas tempranas de saturación
- Análisis de tendencias

## Archivos de Salida

El programa genera automáticamente:

- `histogramas_tiempos_servicio.png`: Distribuciones de tiempos
- `comparacion_metricas.png`: Comparación entre franjas
- `qq_plots_exponencial.png`: Verificación estadística
- `analisis_eficiencia.png`: Análisis de eficiencia

## Troubleshooting

### Errores Comunes

1. **Archivo CSV no encontrado**: Verificar que `datos_9am.csv` y `datos_4pm.csv` existan
2. **Columna faltante**: Asegurar que existe la columna `tiempo_segundos`
3. **Datos no numéricos**: Verificar que todos los valores sean números válidos
4. **Sistema inestable**: ρ ≥ 1 indica que el sistema está saturado

### Validación de Datos

```python
# Verificar estructura de datos
import pandas as pd
datos = pd.read_csv('datos_9am.csv')
print(datos.head())
print(datos.dtypes)
print(datos.isnull().sum())
```
