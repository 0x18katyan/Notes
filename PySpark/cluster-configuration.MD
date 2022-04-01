# Cluster Configurations

Check for running configurations using `spark.conf.get(conf_name)`

```Python
app_name = spark.conf.get('spark.app.name')

port = spark.conf.get('spark.driver.port')
```

Set configurations using `spark.conf.set(conf_name)`

```Python
## Set spark to use 400 partitions
spark.conf.set('spark.sql.shuffle.partitions', 400)
```

## Cluster Types

1. Single Node
2. Multiple Node
3. Managed
    - Yarn
    - Kubernetes
    - Mesos

## Driver

- Task assignment
- Result consolidation
- Shared data access

### Tips

- Driver should have 2x memory of the worker.
- Fast local storage is very helpful.

## Worker

- Runs actual task.
- Ideally has all code, data, and resources for a given task.

### Recommendations

- More workers nodes is often better than less but larger worker nodes.
- Test for balance between performance and expense.

## Performance Improvements

View query plan with `df.explain()`.

## Shuffling

Shuffling is moving data around to and fro various workers to complete a given task.

- It hides the complexity from the user.

- If shuffling is being done often, it can slow down the process because a worker is waiting for the data to arrive.

- Necessary but should be minimized.

### Limit Shuffling

1. Limit use of `.repartition(num_partitions)`, requires full re-shuffling
    - Use `.coalesce(num_partitions)` to reduce the number of partitions.

2. Calling `.join()` without consideration may increase shuffling operations and reduce the throughput of the cluster.

3. Use `.broadcast()` to distribute data to each of the workers.
    - Provides a copy of the data to each worker.
    - Prevents unnecessary communication between nodes.
    - Useful when some data is required for each join.
    - Useful when the data being broadcasted is less than the other data.
    - Import : `from pyspark.sql.functions import broadcast`
    - Usage : `combined_df = df1.join(broadcast(df2))`

4. Optimize for what matters; generate value rather than scrutinizing over small details.
