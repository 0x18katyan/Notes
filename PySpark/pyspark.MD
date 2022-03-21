# Big Data

Dataset too complex fot standard data processing. Definition is vague and varied.

## Introduction of Big Data

### 3 V's of Big Data

Volume : Size of the Data

Variety : Types of Data

Velocity: Speed of Data

### Terminology

Clustered Computing : Collection of resources of multiple machines.

Parallel Computing : Simultaneous computation.

Distributed Computing: Nodes that run in parallel.

Batch Processing: Breaking a large job into small batches and executing them independently.

Real-time Processing: Processing Data in Real-time.

## Introduction to PySpark

PySpark is the Python API for Apache Spark. Apache Spark is written in Scala. PySpark API is similar to Pandas and Scikit-learn.

Spark shell is the interactive environment for running Spark jobs. It is helpful for fast prototyping and interacting with data on disk or in memory. Spark shell is available for Scala, Python and R.

### Features of Apache Spark

- Distributed cluster computer framework.
- Efficient in-memory computation.
- Support for Java, Scala, Python, R and SQL.

### Spark Context

Spark Context is an entry point for Spark. Entry point is way to connect to the Spark cluster.

#### View SparkContext Version

```Python
sc.cluster
```

#### Retrieve Python Version of SparkContext

```Python
sc.pythonVer
```

#### Print URL of cluster

Outputs `local` if the cluster is in local mode.

```Python
sc.master
```

#### Loading Data

##### parallelize() method

```Python
rdd = sc.parallelize([1, 2, 3, 4, 5])
```

##### textFile() method

```Python
rdd2 = sc.textFile("test.txt")
```

## Functional Programming in Python

### Anonymous Functions

- Lambda functions are anonymous functions in Python.

- Efficient, and usually used in conjunction, with map and filter.

- Lambda functions can be called later similar to def.

- It returns functions without any name and has to be assigned to variable to save for reuse.

- There is no return in a lambda function.

- Useful for short and concise operations.

#### lambda definition

`lambda arguments: expression`

Example:

```Python
double = lambda x: x * 2
print(double(2))
```

#### Use of lambda function in filter()

- filter() function takes a function and a list and returns a new list for which the function evaluates as true.

- General Syntax - `filter(function, list)`

- Example:

    ```Python
    items = [1, 2, 3, 4]
    list(filter(lambda x: x%2 != 0, items))
    ```

## Resilient Distributed Datasets (RDD)

- Resilient - Can self-heal in case of corrupted data.

- Distributed - Is distributed across various machines.

- Datasets - Data is partitioned.

### Creating RDDs

- Parallelizing an existing collection of objects.

- External datasets:
    1. Files in HDFS.
    2. Objects in Amazon S3 bucket.
    3. Lines in a text file.

- From existing RDDs.

#### Examples

- parallelize() method:

```Python
rdd = sc.parallelize([1, 2, 3, 4])
rdd2 = sc.parallelize(["Hello World!"])
```

- textFile() method

```Python
fileRDD = sc.textFile(filePath)
```

### Data Partitioning

- Logical Division of a large distributed dataset.

- `min_partitions` can be used to control the number of partitions to use for a dataset.

```Python
rdd = sc.parallelize([1, 2, 3, 4], min_partitions = 3)
rdd2 = sc.textFile(filePath, min_partitions = 3)
```

- `getNumPartitions()` method can be used to check number of partitions of a RDD.

```Python
print(RDD.getNumPartitions())
```

### Operations on RDDs

- Transformations create new RDDs.

- Actions perform computation on the RDDs.

Lazy evaluation is when PySpark builds a DAG by logging all the operations performed, then executes the DAG in a non-blocking manner when any resultant data is required and explicitly returned.

#### Transformations

Basic RDD Transformations - map(), filter(), flatMap(), and union()

- map() Transformation - Applies a function to all the elements in a RDD.

```Python
RDD = sc.parallelize([1, 2, 3, 4])
RDD_map = RDD.map(lambda x: x ** 2)
```

- filter() Transformation - Returns only values that return true.

```Python
RDD = sc.parallelize([1, 2, 3, 4])
RDD_filter = RDD.filter(lambda x: x > 2)
```

- flatMap() Transformation - Returns multiple values from function application.

```Python
RDD = sc.parallelize("This is sorta boring.")
RDD_flatMap = RDD.flatmap(lambda x: x.split(" "))
```

- union() Transformation - Returns the union of two RDDs.

```Python
inputRDD = sc.textFile("logs.txt")

errorRDD = inputRDD.filter(lambda x: "warning" in x.split())
warningsRDD = inputRDD.filter(lambda x: "error" in x.split())
combinedRDD = errorRDD.union(warningsRDD)
```

#### Actions

- Operation return a value after running computation on the RDD.

- Basic RDD Actions:
    1. `collect()` - Returns all elements of the dataset as an array. Example: `RDD.collect()`

    2. `take(N)` - Returns the first *N* elements of the dataset.
    Example: `RDD.take(5)`

    3. `first()` - Returns the first element of the RDD.
    Example: `RDD.first()`

    4. `count()` - Counts the number of elements in a RDD.
    Example: `RDD.count()`

### Pair RDDs in PySpark

Could be understood as dataset API for a Python dictionary.

- Real life datasets are usually key/value pairs.

- Each row is a key and maps to one or more values.

Can be initialized from an array of tuples as :

```Python
some_tuple = [('A', 32), ('B', 68), ('C', 69)] 
pairRDD = sc.parallelize(some_tuple)

normalRDD = [('A 32'), ('B 68'), ('C 69')]
pairRDD_2 = normalRDD.map(lambda s: (s.split(' ')[0], s.split(' ')[1]))
```

- All regular transformations are available for pair RDDs

### Special Transformations for Pair RDD:

There are some special transformations for Pair RDDs.

1. reduceByKey()
2. sortByKey()
3. groupByKey()
4. join()

#### reduceByKey()

- Combines values with the same key.
- Runs parallel operations for each key in the dataset.
- Is a transformation and not an action.

```Python
pairRDD_reduced = regularRDD.reduceByKey(lambda x, y: x * y)
pairRDD_reduced.collect()
```

#### sortByKey()

- Sorts pair RDD by key.

```Python
pairRDD.sortByKey(ascending = False).collect()
```

#### groupByKey()

- groupByKey() groups values with the same key.

```Python
pairRDD_group = regularRDD.groupByKey().collect()

for col1, col2 in pairRDD_group:
    print(col1, list(col2))
```

#### join()

```Python
RDD1.join(RDD2).collect()
```

### Advanced RDD Actions

- saveAsTextFile() action: saves RDD as text file.

- Pair RDD Action:
    1. countByKey() - count number of values by key name.
    2. collectAsMap() - return the key-value pairs as a dictionary.

```Python
total = Rdd.countByKey()

for k, v in total.items():
    print(f"key {k} has {v} counts.")
```

### PySpark DataFrames

- DataFrame is an immutable distributed collection of data with named columns.

- DataFrame API is available in Python, R, Scala, and Java.

- DataFrames support SQL syntax as well as expression methods such as `dataframe.select()`.

- SparkSession is the Entrypoint for the DataFrame API.

- SparkSession is available in PySpark shell as Spark.

- Can be initialized with `RDD.createDataFrame()`.

- Can be initialized with SparkSession's read method.

- Spark DataFrames need schema.

```Python
spark_df = spark.createDataFrame(RDD, schema = column_names)
```

- Schema can be inferred using `inferSchema` option while creating the dataframe.

#### Read Methods

- `spark.read.csv()`
- `spark.read.json()`
- `spark.read.txt()`

