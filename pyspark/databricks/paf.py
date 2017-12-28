from pyspark.sql.functions import udf

sev = spark.read.format('csv')\
           .options(header='true', inferSchema='true')\
           .load("dbfs:/mnt/spark.example/scalars/sev.csv")\
           .withColumnRenamed("value", "sev")
rrmax = spark.read.format('csv')\
             .options(header='true', inferSchema='true')\
             .load("dbfs:/mnt/spark.example/scalars/rrmax.csv")\
             .withColumnRenamed("rr_0", "rrmax")
sev_rrmax = sev.join(rrmax, ['age_group_id', 'sex_id'], 'outer')
# Calculate scalar and paf columns.
sev_rrmax = sev_rrmax.withColumn("scalar", sev_rrmax.sev * (sev_rrmax.rrmax - 1) + 1)\
                     .withColumn("paf", 1 - 1.0/sev_rrmax.scalar)\
                     .dropna()
tmp = sev_rrmax.select('sev', 'rrmax')
# Drop sev, rrmax columns and sort the data.
sev_rrmax = sev_rrmax.drop('sev', 'rrmax')\
                     .orderBy(sev_rrmax.location_id.desc())\
                     .dropDuplicates()
# Use udf to test if sev is above 0.99.
f = udf(lambda x: True if x > 0.99 else False)
tmp = tmp.withColumn('sev_above_threshold', f(tmp.sev))
