import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, sum as _sum, desc, month, year, concat_ws, avg, min as _min, max as _max

# 1. Inicializar sesión de Spark
spark = SparkSession.builder \
    .appName("Taller_Practico_PySpark") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Rutas de archivos
DATA_PATH = "data/Online_Retail.csv"
OUTPUT_DIR = "src/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Cargar y limpiar Dataset
print("Cargando dataset...")
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

df_clean = df.filter(col("InvoiceNo").isNotNull() & col("Quantity").isNotNull() & col("UnitPrice").isNotNull()) \
             .withColumn("TotalAmount", col("Quantity") * col("UnitPrice"))

df_clean.cache()

print("\n==========================================")
print("   RESULTADOS DEL TALLER PYSPARK - ETL   ")
print("==========================================\n")

# Q1: Facturas únicas
total_invoices = df_clean.select(countDistinct("InvoiceNo")).collect()[0][0]
df_q1 = spark.createDataFrame([("Facturas Unicas", total_invoices)], ["Metrica", "Valor"])

# Q2: Clientes únicos
total_customers = df_clean.filter(col("CustomerID").isNotNull()).select(countDistinct("CustomerID")).collect()[0][0]
df_q2 = spark.createDataFrame([("Clientes Unicos", total_customers)], ["Metrica", "Valor"])

# Q3: Ingreso total
total_revenue = df_clean.select(_sum("TotalAmount")).collect()[0][0]
df_q3 = spark.createDataFrame([("Ingreso Total", total_revenue)], ["Metrica", "Valor"])

# Q4: Producto más vendido en cantidad (JOIN)
df_quantities = df_clean.groupBy("StockCode").agg(_sum("Quantity").alias("TotalQuantity"))
df_descriptions = df_clean.select("StockCode", "Description").dropDuplicates(["StockCode"])
df_q4 = df_quantities.join(df_descriptions, "StockCode").orderBy(desc("TotalQuantity"))

# Q5: Cliente con mayor volumen de compra
df_q5 = df_clean.filter(col("CustomerID").isNotNull()) \
                .groupBy("CustomerID") \
                .agg(_sum("TotalAmount").alias("TotalSpentCustomer")) \
                .orderBy(desc("TotalSpentCustomer"))

# Q6: Top 5 países (excluyendo UK)
df_q6 = df_clean.filter(col("Country") != "United Kingdom") \
                .groupBy("Country") \
                .agg(_sum("TotalAmount").alias("TotalSpentCountry")) \
                .orderBy(desc("TotalSpentCountry"))

# Q7: Ticket promedio por factura
df_invoice_metrics = df_clean.groupBy("InvoiceNo") \
                             .agg(_sum("TotalAmount").alias("InvoiceTotal"),
                                  _sum("Quantity").alias("InvoiceProducts"))
avg_ticket = df_invoice_metrics.select(avg("InvoiceTotal")).collect()[0][0]
df_q7 = spark.createDataFrame([("Ticket Promedio", avg_ticket)], ["Metrica", "Valor"])

# Q8: Métricas de productos por factura
df_q8 = df_invoice_metrics.select(
    _min("InvoiceProducts").alias("MinProducts"),
    _max("InvoiceProducts").alias("MaxProducts"),
    avg("InvoiceProducts").alias("AvgProducts")
)

# Q9: Mes con mayor volumen de ventas
df_q9 = df_clean.withColumn("YearMonth", concat_ws("-", year("InvoiceDate"), month("InvoiceDate"))) \
                .groupBy("YearMonth") \
                .agg(_sum("TotalAmount").alias("MonthlySales")) \
                .orderBy(desc("MonthlySales"))

# Q10: Porcentaje de facturas con devoluciones
total_returns = df_clean.filter(col("InvoiceNo").startswith("C")).select(countDistinct("InvoiceNo")).collect()[0][0]
pct_returns = (total_returns / total_invoices) * 100
df_q10 = spark.createDataFrame([("Porcentaje Devoluciones", pct_returns)], ["Metrica", "Valor"])

# Print de verificación en consola
print(f"1. Facturas únicas: {total_invoices}")
print(f"2. Clientes únicos: {total_customers}")
print(f"3. Ingreso total: ${total_revenue:,.2f}")
print("4. Top Productos:")
df_q4.show(1, truncate=False)
print("5. Top Cliente:")
df_q5.show(1, truncate=False)
print("6. Top Países:")
df_q6.show(5, truncate=False)
print(f"7. Ticket promedio: ${avg_ticket:,.2f}")
print("8. Métricas productos:")
df_q8.show()
print("9. Mes top ventas:")
df_q9.show(1)
print(f"10. % Devoluciones: {pct_returns:.2f}%\n")

# ==========================================
# EXPORTACIÓN COMPLETA DE LAS 10 CONSULTAS
# ==========================================
print("Guardando las 10 consultas en src/outputs/...")

df_q1.toPandas().to_csv(f"{OUTPUT_DIR}/q1_total_facturas.csv", index=False)
df_q2.toPandas().to_csv(f"{OUTPUT_DIR}/q2_total_clientes.csv", index=False)
df_q3.toPandas().to_csv(f"{OUTPUT_DIR}/q3_ingreso_total.csv", index=False)
df_q4.limit(10).toPandas().to_csv(f"{OUTPUT_DIR}/q4_top_productos.csv", index=False)
df_q5.limit(10).toPandas().to_csv(f"{OUTPUT_DIR}/q5_top_clientes.csv", index=False)
df_q6.limit(10).toPandas().to_csv(f"{OUTPUT_DIR}/q6_top_paises.csv", index=False)
df_q7.toPandas().to_csv(f"{OUTPUT_DIR}/q7_ticket_promedio.csv", index=False)
df_q8.toPandas().to_csv(f"{OUTPUT_DIR}/q8_metricas_productos.csv", index=False)
df_q9.limit(12).toPandas().to_csv(f"{OUTPUT_DIR}/q9_ventas_mensuales.csv", index=False)
df_q10.toPandas().to_csv(f"{OUTPUT_DIR}/q10_porcentaje_devoluciones.csv", index=False)

print("¡Las 10 consultas se han exportado correctamente!")

spark.stop()