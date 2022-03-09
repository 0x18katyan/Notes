# SQL

## General Operations

### Create Table

Create table `table_name` with 2 columns called `col_1` and `col_2` with `INT32` and `VARCHAR128` data types.

```SQL
CREATE TABLE table_name (
  col_1 INT(32),
  col_2 VARCHAR(128);
```

### Add Column

Add Column that cannot be null and has a default value of 1.

```SQL
ALTER TABLE table_name
ADD column_name int NOT NULL DEFAULT(1);

```

### Delete Column

Drop 2 columns from table.

```SQL
ALTER TABLE table_name
DROP COLUMN column_name
DROP COLUMN column_name2;
```

### View Table Schema

View details about columns in a table called table_name.

```SQL
SELECT * from information_schema.columns where table_name = 'table_name';
```

## Foreign Key

### Add Foreign Key

Add foreign key that references a column from another table.

```SQL
ALTER TABLE source_table_name ADD CONSTRAINT constraint_name
FOREIGN KEY (source_foreign_key) REFERENCES target_table_name(target_column_name)
```

```SQL
ALTER TABLE dim_country_sf ADD CONSTRAINT country_continent
    FOREIGN KEY (continent_id) REFERENCES dim_continent_sf(continent_id);
```

## Join

```SQL
SELECT state, SUM(sales_amount)
FROM fact_booksales
-- Join to get book information
    JOIN dim_book_star on fact_booksales.book_id = dim_book_star.book_id
-- Join to get store information
    JOIN dim_store_star on fact_booksales.store_id = dim_store_star.store_id
-- Get all books with in the novel genre
WHERE  
    dim_book_star.genre = 'novel'
-- Group results by state
GROUP BY
    state;
```

You don't need to specify the table name after executing JOIN statements since it is already joined.

```SQL
-- Output each state and their total sales_amount
SELECT state, SUM(sales_amount)
FROM fact_booksales
    -- Joins for genre
    JOIN dim_book_sf on fact_booksales.book_id = dim_book_sf.book_id  
    JOIN dim_genre_sf on dim_book_sf.genre_id = dim_genre_sf.genre_id
    -- Joins for state 
    JOIN dim_store_sf on fact_booksales.store_id = dim_store_sf.store_id 
    JOIN dim_city_sf on dim_store_sf.city_id = dim_city_sf.city_id
    JOIN dim_state_sf on  dim_city_sf.state_id = dim_state_sf.state_id
-- Get all books with in the novel genre and group the results by state
WHERE  
    dim_genre_sf.genre = 'novel'
GROUP BY
    state;
```

While performing joins, the order of the table.column that are of either sides of `ON` doesn't matter.

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

## Views

A view is the result of a stored query on the data.

1. Query is stored in the memory.
2. Data is aggregated from data in tables.
3. Can be queried like a regular database table.
4. No need to type common queries or alter schemas.

Views can be used to mask the complexities of a convoluted query.


### Create View

Add a line before any query to create a view.

```SQL
CREATE VIEW view_name AS
SELECT col1, col2
FROM table_name
WHERE condition;
```

### View views (PostgreSQL)

To inspect views. The query below also returns system views that come preinstalled with the DBMS.

```SQL
SELECT * FROM INFORMATION_SCHEMA.views;
```

To exclude system views:

```SQL
SELECT * from information_schema.views
WHERE table_schema NOT IN ('pg_catalog', 'information_schema');
```

### View Access Control

Can grant or revoke commands for users on views.

Not all views are updatable. Example: Aggregations.

AVOID MODIFYING DATA FROM VIEWS.

```SQL
-- Revoke everyone's update and insert privileges
REVOKE UPDATE, INSERT ON view_name FROM PUBLIC; 

-- Grant the editor update and insert privileges 
GRANT UPDATE, INSERT ON view_name TO editor; 
```

### Drop View

```SQL
DROP VIEW viewname [CASCADE| RESTRICT];
```

RESTRICT (default): returns an error if there are objects that depend on the view.

CASCADE: drops view and any object that depend on the view.

### Redefine View

Views can be redefined if the data types of the underlying columns or the already present columns are left unchanged and the new column appears at the last of the view.

```SQL
-- Redefine the artist_title view to have a label column
CREATE OR REPLACE VIEW artist_title AS
SELECT reviews.reviewid, reviews.title, artists.artist, labels.label
FROM reviews
INNER JOIN artists
ON artists.reviewid = reviews.reviewid
INNER JOIN labels
ON labels.reviewid = reviews.reviewid;

SELECT * FROM artist_title;
```

### Materialized Views

Materialized views stores the results and not the query.
Materialized views are refreshed or rematerialized when prompted or scheduled.

Materialzed views are useful when the queries take a long time to compute. They can be scheduled during off hours to generate the data which can be later used at convenience. They are mostly used when the underlying isn't constantly changing such as in a Data Warehouse (OLAP).

#### Create Materialized Views

```SQL
CREATE MATERIALIZED VIEW some_view AS SELECT * from table;
```

#### Refresh Materialized Views

```SQL
REFRESH MATERIALIZED VIEW some_view;
```

Since materialized views can have multiple dependencies, an external scheduler or dependency manager such as Apache AirFlow is needed to schedule and run a refresh.
