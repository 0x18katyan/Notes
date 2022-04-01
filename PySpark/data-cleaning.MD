# Data Cleaning in Spark with PySpark

## Why Spark?

Preparing raw data for using in data processing pipelines.

Tasks include:
    - Reformatting text; text processing.
    - Performing calculations; mean, sum, etc.
    - Removing null values.
    - Imputing null values with median.

Problems with conventional data systems:

    - Slow.
    - Not designed for scalability.
    - Organization and optimization of data flow.

## Schema

### Import schema

```Python

from pyspark.sql.types import *

exSchema = StructType([
    ## FieldType(field_name, data_type, allow_null)
    StructField('field1', StringType(), True),
    StructField('field2', IntegerType(), True)
])

```

### Read CSV using schema

```Python
df = spark.read.csv(csv_file_path, schema=exSchema)
```

## Parquet File Format

Parquet is an open source file format which used a columnar storage mechanism for storing data. Being a columnar storage, parquet speeds up various transformations and actions for big data processing engines such as Spark by only executing operations on columns which are needed for the operation.

Parquet also stores the schema with the data itself, which makes it easier for the loading data since the schema doesn't have to be defined every time the data is loaded.

### Syntax

```Python
## Writing a DataFrame to a Parquet File
df.write.parquet('df.parquet', mode = 'overwrite')

## Reading Parquet File
newdf = spark.read.parquet('df.parquet')
```

## Column Operations

```Python
## Select a column
df.select('columnName').show()

## Filter values in a column
df = df.filter('length(columnName) > 0 and length(columnName) < 15')

## Filter using tilde
import pyspark.sql.functions as F

df = df.filter(~ F.col('columnName').contains('shit'))
```

## Built-in Conditionals

`.when()` and `otherwise()` methods provide an efficient way for using conditionals in Spark. The regular approach works too but it is slower than the built-in methods since each row has to be evaluated in the conventional `if-then-else` statement.

Syntax - `when(<if condition>, <then value>)`

```Python
## If statement
df.select(df.Name, df.Age, F.when(df.Age >= 18, "Adult"))

## If and If statement
df.select(df.Name, df.Age, F.when(df.Age >= 18, "Adult").when(df.age < 18, "Minor"))

## If and else statement, only one otherwise in a when chain
df.select(df.Name, df.Age, F.when(df.Age >= 18, "Adult").otherwise("Minor"))
```

## User Defined Functions or UDFs

- Python method.
- Wrapper in `pyspark.sql.functions.udf` method.
- Is stored as a variable.
- Can be called as a normal PySpark function.

```Python
def reverse(val):
    return val[::1]

## Pass the function as variable and expected return value type
reverseUDF = F.udf(reverse, StringType())

df = df.withColumn('reversedName', reverseUDF(df['Name']))
```

## Spark Fundamentals

### Partitioning

- DataFrames are broken up into partitions.
- Partition size can vary.
- Each partition is handled independently.

### Lazy Processing

- Transformations are lazy:
    1. .withColumn()
    2. .select()

- Nothing is done until an action is performed:
    1. .count()
    2. .write()

- Transformations can be re-ordered for best performance but it might lead to expected behavior due to dependencies.

### Caching

- Stores DataFrames in memory or on disk.
- 

```Python
df = df.cache()
```

### File Reading

- More files better than one large file.
- Split files then save as parquet.
- Splitted files offer better performance due to internal optimizations by PySpark.