# Views

A view is the result of a stored query on the data.

1. Query is stored in the memory.
2. Data is aggregated from data in tables.
3. Can be queried like a regular database table.
4. No need to type common queries or alter schemas.

Views can be used to mask the complexities of a convoluted query.

## Non Materialized Views

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

## Materialized Views

Materialized views stores the results and not the query.
Materialized views are refreshed or rematerialized when prompted or scheduled.

Materialzed views are useful when the queries take a long time to compute. They can be scheduled during off hours to generate the data which can be later used at convenience. They are mostly used when the underlying isn't constantly changing such as in a Data Warehouse (OLAP).

### Create Materialized Views

```SQL
CREATE MATERIALIZED VIEW some_view AS SELECT * from table;
```

### Refresh Materialized Views

```SQL
REFRESH MATERIALIZED VIEW some_view;
```

Since materialized views can have multiple dependencies, an external scheduler or dependency manager such as Apache AirFlow is needed to schedule and run a refresh.
