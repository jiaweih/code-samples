from pyspark.sql import SQLContext
from pyspark.sql.functions import col, asc
from pyspark.sql.functions import mean


sqlContext = SQLContext(sc)
diamonds = sqlContext.read.format('csv')\
                     .options(header='true', inferSchema='true')\
                     .load('/databricks-datasets/Rdatasets/data-001/'\
                           'csv/ggplot2/diamonds.csv')
# diamonds.filter(diamonds['color'] == 'E')
diamonds.filter((col("color") == 'E') & (col("cut") == 'Ideal'))\
        .select("carat", "cut", "color", "price")\
        .sort(asc("price"))\
        .take(5)
diamonds.where(col("cut").isNull())
mean_price = diamonds.groupBy("cut", "color").agg(mean('price')).first()
# Returns a new DataFrame by renaming an existing column
mean_price = mean_price.withColumnRenamed("avg(price)", "mean_price")
# Returns a new DataFrame by adding a column or replacing the existing column
# that has the same name.
mean_price = mean_price.withColumn('price_limit', mean_price.mean_price + 2)
mean_price.filter("color == 'E'")\
          .filter("cut == 'Ideal'")\
          .select("carat", "cut", "color", "price")\
          .write.save("/dbfs/mnt/spark.example/diamonds")
# Local Tables: You can create a local table on a cluster that is not
# accessible from other clusters and is not registered in the Hive metastore.
# These are also known as temporary tables.
# To create a local table from a Dataframe in Scala or Python in Spark 2.0:
mean_price.createOrReplaceTempView("mean_price")
diamonds_mean_price = sqlContext.sql("SELECT * FROM mean_price")
