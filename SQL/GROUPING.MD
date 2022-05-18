# GROUP BY

`GROUP BY` is used to segregate the data into groups which allows us to  use different aggregation functions on those specific groups.

```SQL
-- Select the director column
SELECT 
  director,
  -- Sum the earnings column
  SUM(earnings) AS total_earnings
FROM 
  movies
  -- Exclude NULL values of total_earnings
WHERE 
  earnings IS NOT NULL
  -- Group the results by director
GROUP BY 
  director
  -- Order the results in descending order of total_earnings
ORDER BY 
  total_earnings DESC;
```

The `HAVING` clause can be used to filter values of groups or aggregations.

```SQL
-- Select the director column
SELECT 
  director,
  -- Sum the earnings column
  SUM(earnings) AS total_earnings
FROM 
  movies
  -- Exclude NULL values of total_earnings
WHERE 
  earnings IS NOT NULL
  -- Group the results by director
GROUP BY 
  director
  -- Order the results in descending order of total_earnings
HAVING
    SUM(earnings) > 25
ORDER BY 
  total_earnings DESC;
```
