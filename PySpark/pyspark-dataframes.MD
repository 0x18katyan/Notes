# PySpark DataFrames

- DataFrame is an immutable distributed collection of data with named columns.

- DataFrame API is available in Python, R, Scala, and Java.

- DataFrames support SQL syntax as well as expression methods such as `dataframe.select()`.

- SparkSession is the Entrypoint for the DataFrame API.

- SparkSession is available in PySpark shell as Spark.

- Can be initialized with `RDD.createDataFrame()`.

- Can be initialized with SparkSession's read method.

- Spark DataFrames need schema.

## Initializing DataFrames

```Python
spark_df = spark.createDataFrame(RDD, schema = column_names)
```

- Schema can be inferred using `inferSchema` option while creating the dataframe.

### Read Methods

- `spark.read.csv()`
- `spark.read.json()`
- `spark.read.txt()`

```Python
df = spark.read.csv(csv_path, header=True, inferSchema=True)
```

## DataFrame Operations

### Transformations

- select() - subsets the dataframe, takes column names as argument.

    Syntax: `df_col = df.select(col)`

- filter() - subsets the data based on a condition.

    Syntax: `df_fil = df.filter(df.col > 10)`

- groupBy() - groups dataframe by columns.

    Syntax: `df.groupBy(col).sum().show(3)`

- orderBy() - return dataframe sorted by the given column.

    Syntax: `df_sorted_by_col = df.orderBy(col)`

- dropDuplicates() - drops duplicate rows in a dataframe.

    Syntax: `df = df.dropDuplicates()`

- withColumnRenamed() - returns a dataframe with renamed column. 

    Syntax: `df = df.withColumnRenamed(oldColName, newColName)`

### Actions

- printSchema() - prints the column names and their assigned types.

    Syntax: `df.printSchema()`

- head() - print n rows from the top of the dataframe.

    Syntax: `df.head(10)`

- show() - prints the first 20 rows as default.

    Syntax: `df_col.show(3)`

- count() - returns the total number of rows in the dataframe.

    Syntax: `df.count()`

- columns 0 returns a list of the column names.

    Syntax: `df.columns`

- describe(): returns summary statistics for the numerical columns such as mean m median, and standard deviation.

    Syntax: `df.describe().show()`

