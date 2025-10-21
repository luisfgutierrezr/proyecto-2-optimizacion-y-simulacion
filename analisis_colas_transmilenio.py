"""
Análisis de Colas TransMilenio - Modelo M/M/1
Autor: Análisis de Teoría de Colas
Proyecto: Optimización y Simulación - PUJ

Este script analiza los datos de observación de una taquilla de TransMilenio
en dos franjas horarias diferentes (9 AM y 4 PM) utilizando teoría de colas M/M/1.

El modelo M/M/1 asume:
- M: Llegadas con distribución de Poisson
- M: Tiempos de servicio con distribución exponencial
- 1: Un solo servidor
"""

# Importación de librerías necesarias para el análisis
import pandas as pd  # Para manipulación y análisis de datos
import numpy as np  # Para operaciones matemáticas y estadísticas
import matplotlib.pyplot as plt  # Para creación de gráficos
import seaborn as sns  # Para visualizaciones estadísticas avanzadas
from scipy import stats  # Para pruebas estadísticas y distribuciones
import simpy  # Para simulación de sistemas discretos
import warnings
warnings.filterwarnings('ignore')  # Suprimir advertencias no críticas

# Configuración de estilo para gráficas
plt.style.use('seaborn-v0_8')  # Estilo moderno para los gráficos
sns.set_palette("husl")  # Paleta de colores armoniosa

class AnalisisColasTransmilenio:
    """
    Clase principal para el análisis de colas en TransMilenio
    
    Esta clase implementa un análisis completo de teoría de colas para evaluar
    el rendimiento de una taquilla de TransMilenio en dos franjas horarias diferentes.
    Utiliza el modelo M/M/1 para calcular métricas de rendimiento y generar
    recomendaciones de mejora.
    """
    
    def __init__(self):
        self.datos_9am = None  # Almacenará los datos de 9 AM
        self.datos_4pm = None  # Almacenará los datos de 4 PM
        self.resultados_9am = {}  # Resultados del análisis para 9 AM
        self.resultados_4pm = {}  # Resultados del análisis para 4 PM
        
    def cargar_datos(self):
        """
        Cargar datos de ambas franjas horarias desde archivos CSV
        
        Returns:
        - tuple: (datos_9am, datos_4pm) DataFrames con los datos cargados
        
        Los archivos CSV deben contener al menos una columna 'tiempo_segundos'
        con los tiempos de servicio observados para cada cliente.
        """
        print("Cargando datos de observación...")
        
        # Cargar datos de la franja horaria de 9 AM
        # Se espera un archivo CSV con tiempos de servicio en segundos
        self.datos_9am = pd.read_csv('datos_9am.csv')
        print(f"Datos 9 AM: {len(self.datos_9am)} observaciones")
        
        # Cargar datos de la franja horaria de 4 PM
        # Se espera un archivo CSV con tiempos de servicio en segundos
        self.datos_4pm = pd.read_csv('datos_4pm.csv')
        print(f"Datos 4 PM: {len(self.datos_4pm)} observaciones")
        
        return self.datos_9am, self.datos_4pm
    
    def calcular_metricas_basicas(self, datos, franja_horaria):
        """
        Calcular métricas básicas para una franja horaria usando el modelo M/M/1
        
        Args:
        - datos: DataFrame con los datos de observación
        - franja_horaria: String identificando la franja horaria
        
        Returns:
        - dict: Diccionario con todas las métricas calculadas
        
        Métricas calculadas:
        - L: Número promedio de clientes en el sistema
        - Lq: Número promedio de clientes en la cola
        - W: Tiempo promedio en el sistema
        - Wq: Tiempo promedio en la cola
        - ρ (rho): Factor de utilización del servidor
        """
        # Extraer tiempos de servicio de los datos
        tiempos_servicio = datos['tiempo_segundos'].values
        
        # Calcular estadísticas descriptivas básicas
        media_servicio = np.mean(tiempos_servicio)  # Tiempo promedio de servicio
        desv_servicio = np.std(tiempos_servicio)     # Desviación estándar
        
        # Parámetros del sistema
        tiempo_total = 600  # 10 minutos de observación = 600 segundos
        num_clientes = len(tiempos_servicio)  # Total de clientes observados
        
        # Calcular tasas del sistema
        tasa_llegada = num_clientes / tiempo_total  # λ (lambda): tasa de llegada
        tasa_servicio = 1 / media_servicio         # μ (mu): tasa de servicio
        
        # Factor de utilización (ρ = λ/μ)
        # Indica qué porcentaje del tiempo está ocupado el servidor
        rho = tasa_llegada / tasa_servicio
        
        # Calcular métricas teóricas del modelo M/M/1
        # Solo válidas si ρ < 1 (condición de estabilidad)
        if rho < 1:  # Sistema estable
            L = rho / (1 - rho)  # Clientes promedio en el sistema
            Lq = (rho ** 2) / (1 - rho)  # Clientes promedio en cola
            W = 1 / (tasa_servicio - tasa_llegada)  # Tiempo promedio en sistema
            Wq = rho / (tasa_servicio - tasa_llegada)  # Tiempo promedio en cola
        else:
            # Sistema inestable - cola crece indefinidamente
            L = Lq = W = Wq = float('inf')
        
        # Compilar todos los resultados en un diccionario
        resultados = {
            'franja_horaria': franja_horaria,
            'num_clientes': num_clientes,
            'tiempo_total': tiempo_total,
            'media_servicio': media_servicio,
            'desv_servicio': desv_servicio,
            'tasa_llegada': tasa_llegada,
            'tasa_servicio': tasa_servicio,
            'rho': rho,
            'L': L,
            'Lq': Lq,
            'W': W,
            'Wq': Wq,
            'tiempos_servicio': tiempos_servicio
        }
        
        return resultados
    
    def verificar_distribucion_exponencial(self, datos, franja_horaria):
        """
        Verificar si los datos siguen distribución exponencial
        
        Args:
        - datos: DataFrame con los datos de observación
        - franja_horaria: String identificando la franja horaria
        
        Returns:
        - dict: Estadísticas de la prueba de bondad de ajuste
        """
        # Extraer tiempos de servicio
        tiempos = datos['tiempo_segundos'].values
        
        # Ajustar una distribución exponencial a los datos observados
        # loc: parámetro de localización, scale: parámetro de escala
        loc, scale = stats.expon.fit(tiempos)
        
        # Realizar prueba de Kolmogorov-Smirnov para verificar bondad de ajuste
        # H0: Los datos siguen distribución exponencial
        # H1: Los datos NO siguen distribución exponencial
        ks_stat, ks_pvalue = stats.kstest(tiempos, lambda x: stats.expon.cdf(x, loc, scale))
        
        # Calcular estadísticas teóricas de la distribución ajustada
        media_teorica = stats.expon.mean(loc, scale)  # Media teórica
        varianza_teorica = stats.expon.var(loc, scale)  # Varianza teórica
        
        return {
            'ks_statistic': ks_stat,      # Estadístico de la prueba KS
            'ks_pvalue': ks_pvalue,       # p-valor (si > 0.05, no rechazamos H0)
            'expon_loc': loc,            # Parámetro de localización
            'expon_scale': scale,         # Parámetro de escala
            'media_teorica': media_teorica,     # Media de la distribución ajustada
            'varianza_teorica': varianza_teorica # Varianza de la distribución ajustada
        }
    
    def simular_cola_mm1(self, tasa_llegada, tasa_servicio, tiempo_simulacion=600):
        """
        Simular sistema M/M/1 usando SimPy
        
        Esta función implementa una simulación discreta del sistema de colas M/M/1
        para validar los resultados teóricos obtenidos con las fórmulas analíticas.
        
        Args:
        - tasa_llegada: Tasa de llegada de clientes (λ)
        - tasa_servicio: Tasa de servicio del servidor (μ)
        - tiempo_simulacion: Duración de la simulación en segundos
        
        Returns:
        - dict: Parámetros de la simulación ejecutada
        """
        
        def cliente(env, servidor, nombre):
            """
            Proceso que representa el comportamiento de un cliente individual
            
            Cada cliente:
            1. Llega al sistema y registra su tiempo de llegada
            2. Espera en cola hasta que el servidor esté disponible
            3. Recibe servicio durante un tiempo exponencial
            4. Sale del sistema
            """
            llegada = env.now  # Registrar tiempo de llegada
            
            # Solicitar acceso al servidor (cola automática)
            with servidor.request() as req:
                yield req  # Esperar a que el servidor esté disponible
                espera = env.now - llegada  # Calcular tiempo de espera en cola
                
                # Generar tiempo de servicio con distribución exponencial
                # El parámetro es 1/μ para obtener media = 1/μ
                tiempo_servicio = np.random.exponential(1/tasa_servicio)
                yield env.timeout(tiempo_servicio)  # Simular tiempo de servicio
                
                # Retornar estadísticas del cliente
                return {
                    'cliente': nombre,
                    'tiempo_llegada': llegada,
                    'tiempo_espera': espera,
                    'tiempo_servicio': tiempo_servicio,
                    'tiempo_salida': env.now,
                    'tiempo_total': env.now - llegada
                }
        
        def generador_clientes(env, servidor):
            """
            Generador de llegadas de clientes con distribución de Poisson
            
            Los intervalos entre llegadas siguen distribución exponencial,
            lo que genera un proceso de Poisson con tasa λ.
            """
            cliente_id = 0
            while True:
                # Generar tiempo hasta la próxima llegada (exponencial)
                yield env.timeout(np.random.exponential(1/tasa_llegada))
                cliente_id += 1
                # Crear nuevo proceso de cliente
                env.process(cliente(env, servidor, cliente_id))
        
        # Configurar ambiente de simulación
        env = simpy.Environment()  # Crear ambiente de simulación
        servidor = simpy.Resource(env, capacity=1)  # Servidor con capacidad 1
        
        # Iniciar el generador de clientes
        env.process(generador_clientes(env, servidor))
        
        # Ejecutar simulación por el tiempo especificado
        env.run(until=tiempo_simulacion)
        
        return {
            'tiempo_simulacion': tiempo_simulacion,
            'tasa_llegada': tasa_llegada,
            'tasa_servicio': tasa_servicio
        }
    
    def crear_visualizaciones(self):
        """
        Crear todas las visualizaciones del análisis
        
        Genera cuatro tipos de gráficos:
        1. Histogramas de distribución de tiempos de servicio
        2. Comparación de métricas entre franjas horarias
        3. Q-Q plots para verificar distribución exponencial
        4. Análisis de eficiencia del sistema
        """
        
        # 1. HISTOGRAMAS DE TIEMPOS DE SERVICIO
        # Mostrar la distribución empírica de los tiempos de servicio
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histograma para 9 AM
        axes[0].hist(self.resultados_9am['tiempos_servicio'], bins=15, alpha=0.7, 
                    color='skyblue', edgecolor='black')
        axes[0].set_title('Distribución de Tiempos de Servicio - 9 AM', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Tiempo (segundos)')
        axes[0].set_ylabel('Frecuencia')
        axes[0].grid(True, alpha=0.3)
        
        # Histograma para 4 PM
        axes[1].hist(self.resultados_4pm['tiempos_servicio'], bins=15, alpha=0.7, 
                    color='lightcoral', edgecolor='black')
        axes[1].set_title('Distribución de Tiempos de Servicio - 4 PM', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Tiempo (segundos)')
        axes[1].set_ylabel('Frecuencia')
        axes[1].grid(True, alpha=0.3)
        
        # Guardar y mostrar histogramas
        plt.tight_layout()
        plt.savefig('histogramas_tiempos_servicio.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. COMPARACIÓN DE MÉTRICAS ENTRE FRANJAS HORARIAS
        # Crear gráficos de barras para comparar métricas clave
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Configuración para gráficos de comparación
        franjas = ['9 AM', '4 PM']
        colores = ['skyblue', 'lightcoral']
        
        # Definir métricas clave a comparar
        metricas = ['rho', 'L', 'Lq', 'W']
        titulos = ['Factor de Utilización (ρ)', 'Clientes en Sistema (L)', 
                  'Clientes en Cola (Lq)', 'Tiempo en Sistema (W)']
        
        # Extraer valores para cada métrica
        valores_9am = [self.resultados_9am[metrica] for metrica in metricas]
        valores_4pm = [self.resultados_4pm[metrica] for metrica in metricas]
        
        # Crear gráficos de barras para cada métrica
        for i, (metrica, titulo) in enumerate(zip(metricas, titulos)):
            ax = axes[i//2, i%2]  # Posicionar en la cuadrícula 2x2
            
            # Obtener valores para ambas franjas horarias
            valores = [self.resultados_9am[metrica], self.resultados_4pm[metrica]]
            barras = ax.bar(franjas, valores, color=colores, alpha=0.7, edgecolor='black')
            
            # Configurar título y etiquetas
            ax.set_title(titulo, fontsize=12, fontweight='bold')
            ax.set_ylabel('Valor')
            ax.grid(True, alpha=0.3)
            
            # Añadir valores numéricos en las barras para mejor legibilidad
            for barra, valor in zip(barras, valores):
                if valor != float('inf'):  # Evitar mostrar infinito
                    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.01,
                           f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Guardar y mostrar gráficos de comparación
        plt.tight_layout()
        plt.savefig('comparacion_metricas.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Q-Q PLOTS PARA VERIFICAR DISTRIBUCIÓN EXPONENCIAL
        # Los Q-Q plots ayudan a verificar si los datos siguen la distribución asumida
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Q-Q Plot para 9 AM
        # Si los puntos se alinean con la línea diagonal, los datos siguen distribución exponencial
        stats.probplot(self.resultados_9am['tiempos_servicio'], dist="expon", plot=axes[0])
        axes[0].set_title('Q-Q Plot - 9 AM (Distribución Exponencial)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q Plot para 4 PM
        # La proximidad a la línea diagonal indica bondad de ajuste
        stats.probplot(self.resultados_4pm['tiempos_servicio'], dist="expon", plot=axes[1])
        axes[1].set_title('Q-Q Plot - 4 PM (Distribución Exponencial)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Guardar y mostrar Q-Q plots
        plt.tight_layout()
        plt.savefig('qq_plots_exponencial.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. ANÁLISIS DE EFICIENCIA DEL SISTEMA
        # Mostrar la capacidad ociosa del sistema (1 - ρ) como porcentaje
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Calcular eficiencia como porcentaje de capacidad ociosa
        eficiencia_9am = (1 - self.resultados_9am['rho']) * 100
        eficiencia_4pm = (1 - self.resultados_4pm['rho']) * 100
        
        # Configurar datos para el gráfico
        franjas = ['9 AM', '4 PM']
        eficiencias = [eficiencia_9am, eficiencia_4pm]
        # Código de colores: verde (>20%), naranja (10-20%), rojo (<10%)
        colores = ['green' if eff > 20 else 'orange' if eff > 10 else 'red' for eff in eficiencias]
        
        # Crear gráfico de barras con código de colores
        barras = ax.bar(franjas, eficiencias, color=colores, alpha=0.7, edgecolor='black')
        ax.set_title('Eficiencia del Sistema (Capacidad Ociosa)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Eficiencia (%)')
        ax.set_ylim(0, 100)  # Eficiencia como porcentaje
        ax.grid(True, alpha=0.3)
        
        # Añadir valores numéricos en las barras
        for barra, valor in zip(barras, eficiencias):
            ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 1,
                   f'{valor:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Guardar y mostrar análisis de eficiencia
        plt.tight_layout()
        plt.savefig('analisis_eficiencia.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generar_propuesta_mejora(self):
        """Generar propuesta de mejora basada en el análisis"""
        print("\n" + "="*60)
        print("ANÁLISIS Y PROPUESTA DE MEJORA")
        print("="*60)
        
        rho_9am = self.resultados_9am['rho']
        rho_4pm = self.resultados_4pm['rho']
        
        print(f"\nFactor de utilización 9 AM: {rho_9am:.3f}")
        print(f"Factor de utilización 4 PM: {rho_4pm:.3f}")
        
        if rho_4pm > 0.8:
            print("\nALERTA: El sistema de 4 PM está cerca de la saturación (ρ > 0.8)")
            print("RECOMENDACIÓN: Implementar una segunda taquilla durante la hora pico vespertina")
            
            # Calcular impacto de una segunda taquilla (M/M/2)
            nueva_tasa_servicio = self.resultados_4pm['tasa_servicio'] * 2
            nuevo_rho = self.resultados_4pm['tasa_llegada'] / nueva_tasa_servicio
            
            print(f"\nCon una segunda taquilla:")
            print(f"  - Nuevo factor de utilización: {nuevo_rho:.3f}")
            print(f"  - Reducción en utilización: {((rho_4pm - nuevo_rho)/rho_4pm)*100:.1f}%")
            
        elif rho_9am > 0.7:
            print("\nADVERTENCIA: El sistema de 9 AM tiene alta utilización (ρ > 0.7)")
            print("RECOMENDACIÓN: Monitorear y considerar personal adicional durante picos matutinos")
        else:
            print("\nEl sistema actual mantiene niveles aceptables de utilización")
        
        print(f"\nCOMPARACIÓN DE EFICIENCIA:")
        eficiencia_9am = (1 - rho_9am) * 100
        eficiencia_4pm = (1 - rho_4pm) * 100
        
        print(f"  - Eficiencia 9 AM: {eficiencia_9am:.1f}%")
        print(f"  - Eficiencia 4 PM: {eficiencia_4pm:.1f}%")
        
        if eficiencia_4pm < eficiencia_9am:
            print(f"  - La franja de 4 PM es {eficiencia_9am - eficiencia_4pm:.1f}% menos eficiente")
        
        print(f"\nPROPUESTA ESPECÍFICA:")
        print("1. Implementar taquilla adicional de 4:00 PM a 5:00 PM")
        print("2. Capacitar personal para manejo de picos de demanda")
        print("3. Implementar sistema de información en tiempo real sobre tiempos de espera")
        print("4. Considerar automatización parcial para transacciones simples")
    
    def ejecutar_analisis_completo(self):
        """
        Ejecutar el análisis completo del sistema de colas
        
        Este método orquesta todo el proceso de análisis:
        1. Carga los datos de ambas franjas horarias
        2. Calcula métricas para cada franja
        3. Verifica las distribuciones estadísticas
        4. Genera visualizaciones
        5. Produce recomendaciones de mejora
        """
        print("ANÁLISIS DE COLAS TRANSMILENIO - MODELO M/M/1")
        print("="*60)
        
        # PASO 1: Cargar datos de observación
        self.cargar_datos()
        
        # PASO 2: Analizar ambas franjas horarias
        print("\nAnalizando franja de 9 AM...")
        self.resultados_9am = self.calcular_metricas_basicas(self.datos_9am, "9 AM")
        
        print("Analizando franja de 4 PM...")
        self.resultados_4pm = self.calcular_metricas_basicas(self.datos_4pm, "4 PM")
        
        # PASO 3: Mostrar resultados numéricos
        self.mostrar_resultados()
        
        # PASO 4: Verificar asunciones estadísticas del modelo
        print("\nVerificando distribuciones exponenciales...")
        stats_9am = self.verificar_distribucion_exponencial(self.datos_9am, "9 AM")
        stats_4pm = self.verificar_distribucion_exponencial(self.datos_4pm, "4 PM")
        
        # Mostrar resultados de las pruebas de bondad de ajuste
        print(f"Prueba KS 9 AM - Estadístico: {stats_9am['ks_statistic']:.4f}, p-valor: {stats_9am['ks_pvalue']:.4f}")
        print(f"Prueba KS 4 PM - Estadístico: {stats_4pm['ks_statistic']:.4f}, p-valor: {stats_4pm['ks_pvalue']:.4f}")
        
        # PASO 5: Crear visualizaciones
        print("\nGenerando visualizaciones...")
        self.crear_visualizaciones()
        
        # PASO 6: Generar propuesta de mejora
        self.generar_propuesta_mejora()
        
        print(f"\n Análisis completado. Gráficas guardadas en el directorio actual.")
    
    def mostrar_resultados(self):
        """
        Mostrar resultados del análisis en formato tabular
        
        Presenta todas las métricas calculadas para ambas franjas horarias.
        """
        print("\n" + "="*60)
        print("RESULTADOS DEL ANÁLISIS")
        print("="*60)
        
        # Configurar datos para presentación
        franjas = ['9 AM', '4 PM']
        resultados = [self.resultados_9am, self.resultados_4pm]
        
        # Mostrar resultados para cada franja horaria
        for franja, resultado in zip(franjas, resultados):
            print(f"\nFRANJA HORARIA: {franja}")
            print("-" * 30)
            
            # Estadísticas descriptivas
            print(f"Número de clientes observados: {resultado['num_clientes']}")
            print(f"Tiempo promedio de servicio: {resultado['media_servicio']:.2f} segundos")
            print(f"Desviación estándar servicio: {resultado['desv_servicio']:.2f} segundos")
            
            # Parámetros del modelo
            print(f"Tasa de llegada (λ): {resultado['tasa_llegada']:.4f} clientes/seg")
            print(f"Tasa de servicio (μ): {resultado['tasa_servicio']:.4f} clientes/seg")
            print(f"Factor de utilización (ρ): {resultado['rho']:.3f}")
            
            # Mostrar métricas de rendimiento (solo si el sistema es estable)
            if resultado['rho'] < 1:
                print(f"Clientes promedio en sistema (L): {resultado['L']:.3f}")
                print(f"Clientes promedio en cola (Lq): {resultado['Lq']:.3f}")
                print(f"Tiempo promedio en sistema (W): {resultado['W']:.2f} segundos")
                print(f"Tiempo promedio en cola (Wq): {resultado['Wq']:.2f} segundos")
            else:
                print("Sistema inestable (ρ ≥ 1) - Las colas crecen indefinidamente")

def main():
    # Crear instancia del analizador
    analisis = AnalisisColasTransmilenio()
    
    # Ejecutar análisis completo
    analisis.ejecutar_analisis_completo()

if __name__ == "__main__":
    main()
