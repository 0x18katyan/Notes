# Table Partition

Partitioning is physical data model.

## Vertical Partitioning

Splits table vertically by columns. Store less queried columns on slower partitions.

## Horizontal Partitioning

Splits table by rows. Could partition by timestamp so as to keep newer data in faster resources.

```SQL
PARTITION BY RANGE (timestamp);

CREATE TABLE sales_2021_q1 PARTITION OF sales 
    FOR VALUES FROM ('2021-01-01') TO ('2019-03-31');

CREATE INDEX ON sales ('timestamp');
```

### Pros and Cons of Horizontal Partitioning

#### Pros

- Indices of heavily-used partitions fit in memory
- Used for both OLAP and OLTP

#### Cons

- Partitioning existing tables is a hassle.
- Some constraints can't be set up for partitioning.
