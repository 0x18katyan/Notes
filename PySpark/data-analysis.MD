# Notes

- Matplotlib, and it's derivatives, are not supported by PySpark.

- Plotting in PySpark is done by:

    1. Using pyspark_dist_explore library.
        - Currently only has `hist()`, `distplot()` and `pandas_histogram()` plots.

    2. Convert to pandas using `toPandas()` method.
        - Large datasets may not fit into memory for a single machine.

    3. Using HandySpark library.
        - Relatively new.
