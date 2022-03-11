## Snowflake Schema

Updating is usually easier in snowflake schema due to less redundancy. Snowflake schema are also better at naming due to referential integrity.

## Normalization

Normalization is the process of organizing the data in the database. It is used to reduce redundancy. It divides a larger table in smaller tables and links them using relationship.

### Normalization Types

#### 1NF rules

1. Each record must be unique - no dupicate rows.
2. Each cell must hold one value.

#### 2NF rules

1. Must satisfy 1NF AND
    - If primary key is one column then automatically satisfies 2NF.
2. If there is a composite primary key
    - then each non-key column must be dependent on all the keys.

#### 3NF rules

1. Satisfies 2NF.
2. No transitive dependencies: non-key columns can't depend on other non-key columns.

### Why Normalize?

1. Update anomaly - Data inconsistency caysed by data redundancy when updating.

2. Insertion anomaly - Unable to add a record due to missing attributes.

3. Deletion anomaly - Deletion of record(s) causes unintentional loss of data.

### Why not to Normalize?

Normalization means longer queries (because of joins) for fetching the data due to the many relationships between the tables.
