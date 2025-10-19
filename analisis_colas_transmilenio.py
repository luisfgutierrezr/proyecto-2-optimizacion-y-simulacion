"""
Análisis de Colas TransMilenio - Modelo M/M/1
Autor: Análisis de Teoría de Colas
Proyecto: Optimización y Simulación - PUJ

Este script analiza los datos de observación de una taquilla de TransMilenio
en dos franjas horarias diferentes (9 AM y 4 PM) utilizando teoría de colas M/M/1.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import simpy
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficas
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AnalisisColasTransmilenio:
    """Clase principal para el análisis de colas en TransMilenio"""
    
    def __init__(self):
        self.datos_9am = None
        self.datos_4pm = None
        self.resultados_9am = {}
        self.resultados_4pm = {}
        
    def cargar_datos(self):
        """Cargar datos de ambas franjas horarias"""
        print("Cargando datos de observación...")
        
        # Cargar datos de 9 AM
        self.datos_9am = pd.read_csv('datos_9am.csv')
        print(f"Datos 9 AM: {len(self.datos_9am)} observaciones")
        
        # Cargar datos de 4 PM
        self.datos_4pm = pd.read_csv('datos_4pm.csv')
        print(f"Datos 4 PM: {len(self.datos_4pm)} observaciones")
        
        return self.datos_9am, self.datos_4pm
    
    def calcular_metricas_basicas(self, datos, franja_horaria):
        """Calcular métricas básicas para una franja horaria"""
        tiempos_servicio = datos['tiempo_segundos'].values
        
        # Estadísticas descriptivas
        media_servicio = np.mean(tiempos_servicio)
        desv_servicio = np.std(tiempos_servicio)
        
        # Tiempo total de observación (10 minutos = 600 segundos)
        tiempo_total = 600
        num_clientes = len(tiempos_servicio)
        
        # Tasas
        tasa_llegada = num_clientes / tiempo_total  # λ (clientes por segundo)
        tasa_servicio = 1 / media_servicio  # μ (clientes por segundo)
        
        # Factor de utilización
        rho = tasa_llegada / tasa_servicio
        
        # Métricas teóricas M/M/1
        if rho < 1:  # Condición de estabilidad
            L = rho / (1 - rho)  # Clientes promedio en el sistema
            Lq = (rho ** 2) / (1 - rho)  # Clientes promedio en cola
            W = 1 / (tasa_servicio - tasa_llegada)  # Tiempo promedio en sistema
            Wq = rho / (tasa_servicio - tasa_llegada)  # Tiempo promedio en cola
        else:
            L = Lq = W = Wq = float('inf')
        
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
        """Verificar si los datos siguen distribución exponencial"""
        tiempos = datos['tiempo_segundos'].values
        
        # Ajustar distribución exponencial
        loc, scale = stats.expon.fit(tiempos)
        
        # Prueba de Kolmogorov-Smirnov
        ks_stat, ks_pvalue = stats.kstest(tiempos, lambda x: stats.expon.cdf(x, loc, scale))
        
        # Estadísticas para Q-Q plot
        media_teorica = stats.expon.mean(loc, scale)
        varianza_teorica = stats.expon.var(loc, scale)
        
        return {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'expon_loc': loc,
            'expon_scale': scale,
            'media_teorica': media_teorica,
            'varianza_teorica': varianza_teorica
        }
    
    def simular_cola_mm1(self, tasa_llegada, tasa_servicio, tiempo_simulacion=600):
        """Simular sistema M/M/1 usando SimPy"""
        
        def cliente(env, servidor, nombre):
            """Proceso de un cliente"""
            llegada = env.now
            
            with servidor.request() as req:
                yield req  # Esperar a que el servidor esté disponible
                espera = env.now - llegada
                
                # Tiempo de servicio (distribución exponencial)
                tiempo_servicio = np.random.exponential(1/tasa_servicio)
                yield env.timeout(tiempo_servicio)
                
                return {
                    'cliente': nombre,
                    'tiempo_llegada': llegada,
                    'tiempo_espera': espera,
                    'tiempo_servicio': tiempo_servicio,
                    'tiempo_salida': env.now,
                    'tiempo_total': env.now - llegada
                }
        
        def generador_clientes(env, servidor):
            """Generar llegadas de clientes"""
            cliente_id = 0
            while True:
                yield env.timeout(np.random.exponential(1/tasa_llegada))
                cliente_id += 1
                env.process(cliente(env, servidor, cliente_id))
        
        # Crear ambiente de simulación
        env = simpy.Environment()
        servidor = simpy.Resource(env, capacity=1)
        
        # Iniciar generador de clientes
        env.process(generador_clientes(env, servidor))
        
        # Ejecutar simulación
        env.run(until=tiempo_simulacion)
        
        # Recopilar estadísticas (simplificado para esta implementación)
        # En una implementación más completa, se usarían monitores de SimPy
        
        return {
            'tiempo_simulacion': tiempo_simulacion,
            'tasa_llegada': tasa_llegada,
            'tasa_servicio': tasa_servicio
        }
    
    def crear_visualizaciones(self):
        """Crear todas las visualizaciones del análisis"""
        
        # 1. Histogramas de tiempos de servicio
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 9 AM
        axes[0].hist(self.resultados_9am['tiempos_servicio'], bins=15, alpha=0.7, 
                    color='skyblue', edgecolor='black')
        axes[0].set_title('Distribución de Tiempos de Servicio - 9 AM', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Tiempo (segundos)')
        axes[0].set_ylabel('Frecuencia')
        axes[0].grid(True, alpha=0.3)
        
        # 4 PM
        axes[1].hist(self.resultados_4pm['tiempos_servicio'], bins=15, alpha=0.7, 
                    color='lightcoral', edgecolor='black')
        axes[1].set_title('Distribución de Tiempos de Servicio - 4 PM', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Tiempo (segundos)')
        axes[1].set_ylabel('Frecuencia')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('histogramas_tiempos_servicio.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Comparación de métricas
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        franjas = ['9 AM', '4 PM']
        colores = ['skyblue', 'lightcoral']
        
        # Métricas a comparar
        metricas = ['rho', 'L', 'Lq', 'W']
        titulos = ['Factor de Utilización (ρ)', 'Clientes en Sistema (L)', 
                  'Clientes en Cola (Lq)', 'Tiempo en Sistema (W)']
        
        valores_9am = [self.resultados_9am[metrica] for metrica in metricas]
        valores_4pm = [self.resultados_4pm[metrica] for metrica in metricas]
        
        for i, (metrica, titulo) in enumerate(zip(metricas, titulos)):
            ax = axes[i//2, i%2]
            
            valores = [self.resultados_9am[metrica], self.resultados_4pm[metrica]]
            barras = ax.bar(franjas, valores, color=colores, alpha=0.7, edgecolor='black')
            
            ax.set_title(titulo, fontsize=12, fontweight='bold')
            ax.set_ylabel('Valor')
            ax.grid(True, alpha=0.3)
            
            # Añadir valores en las barras
            for barra, valor in zip(barras, valores):
                if valor != float('inf'):
                    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.01,
                           f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('comparacion_metricas.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Q-Q Plots para verificar distribución exponencial
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 9 AM Q-Q Plot
        stats.probplot(self.resultados_9am['tiempos_servicio'], dist="expon", plot=axes[0])
        axes[0].set_title('Q-Q Plot - 9 AM (Distribución Exponencial)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 4 PM Q-Q Plot
        stats.probplot(self.resultados_4pm['tiempos_servicio'], dist="expon", plot=axes[1])
        axes[1].set_title('Q-Q Plot - 4 PM (Distribución Exponencial)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('qq_plots_exponencial.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Análisis de eficiencia
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        eficiencia_9am = (1 - self.resultados_9am['rho']) * 100
        eficiencia_4pm = (1 - self.resultados_4pm['rho']) * 100
        
        franjas = ['9 AM', '4 PM']
        eficiencias = [eficiencia_9am, eficiencia_4pm]
        colores = ['green' if eff > 20 else 'orange' if eff > 10 else 'red' for eff in eficiencias]
        
        barras = ax.bar(franjas, eficiencias, color=colores, alpha=0.7, edgecolor='black')
        ax.set_title('Eficiencia del Sistema (Capacidad Ociosa)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Eficiencia (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for barra, valor in zip(barras, eficiencias):
            ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 1,
                   f'{valor:.1f}%', ha='center', va='bottom', fontweight='bold')
        
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
        """Ejecutar el análisis completo"""
        print("ANÁLISIS DE COLAS TRANSMILENIO - MODELO M/M/1")
        print("="*60)
        
        # Cargar datos
        self.cargar_datos()
        
        # Analizar ambas franjas horarias
        print("\nAnalizando franja de 9 AM...")
        self.resultados_9am = self.calcular_metricas_basicas(self.datos_9am, "9 AM")
        
        print("Analizando franja de 4 PM...")
        self.resultados_4pm = self.calcular_metricas_basicas(self.datos_4pm, "4 PM")
        
        # Mostrar resultados
        self.mostrar_resultados()
        
        # Verificar distribuciones
        print("\nVerificando distribuciones exponenciales...")
        stats_9am = self.verificar_distribucion_exponencial(self.datos_9am, "9 AM")
        stats_4pm = self.verificar_distribucion_exponencial(self.datos_4pm, "4 PM")
        
        print(f"Prueba KS 9 AM - Estadístico: {stats_9am['ks_statistic']:.4f}, p-valor: {stats_9am['ks_pvalue']:.4f}")
        print(f"Prueba KS 4 PM - Estadístico: {stats_4pm['ks_statistic']:.4f}, p-valor: {stats_4pm['ks_pvalue']:.4f}")
        
        # Crear visualizaciones
        print("\nGenerando visualizaciones...")
        self.crear_visualizaciones()
        
        # Generar propuesta de mejora
        self.generar_propuesta_mejora()
        
        print(f"\n Análisis completado. Gráficas guardadas en el directorio actual.")
    
    def mostrar_resultados(self):
        """Mostrar resultados del análisis"""
        print("\n" + "="*60)
        print("RESULTADOS DEL ANÁLISIS")
        print("="*60)
        
        franjas = ['9 AM', '4 PM']
        resultados = [self.resultados_9am, self.resultados_4pm]
        
        for franja, resultado in zip(franjas, resultados):
            print(f"\nFRANJA HORARIA: {franja}")
            print("-" * 30)
            print(f"Número de clientes observados: {resultado['num_clientes']}")
            print(f"Tiempo promedio de servicio: {resultado['media_servicio']:.2f} segundos")
            print(f"Desviación estándar servicio: {resultado['desv_servicio']:.2f} segundos")
            print(f"Tasa de llegada (λ): {resultado['tasa_llegada']:.4f} clientes/seg")
            print(f"Tasa de servicio (μ): {resultado['tasa_servicio']:.4f} clientes/seg")
            print(f"Factor de utilización (ρ): {resultado['rho']:.3f}")
            
            if resultado['rho'] < 1:
                print(f"Clientes promedio en sistema (L): {resultado['L']:.3f}")
                print(f"Clientes promedio en cola (Lq): {resultado['Lq']:.3f}")
                print(f"Tiempo promedio en sistema (W): {resultado['W']:.2f} segundos")
                print(f"Tiempo promedio en cola (Wq): {resultado['Wq']:.2f} segundos")
            else:
                print("Sistema inestable (ρ ≥ 1)")

def main():
    """Función principal"""
    analisis = AnalisisColasTransmilenio()
    analisis.ejecutar_analisis_completo()

if __name__ == "__main__":
    main()
