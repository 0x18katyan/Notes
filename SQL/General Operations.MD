# General Operations

## CRUD

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

### Insert Data

#### Insert from another table

```SQL
INSERT INTO target_table
SELECT column_1, column_2 FROM source_table;
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

