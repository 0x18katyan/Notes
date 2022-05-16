# Writing Queries Properly

## AND OR

`AND` and `OR` clauses can be combined with the `WHERE` clause to filter results using conditionals. 

Note: Common conditions should be in the same parentheses.

```SQL
SELECT song
FROM songs
WHERE
    artist = 'Bob Dylan'
    AND (
        release_year = 1966
        OR release_year < 1976
    );
```

```SQL
SELECT song
FROM songs
WHERE
    (
        artist = 'Bob Dylan'
        AND release_year = 1966
    )
    OR (
        artist = 'Bob Dylan'
        AND release_year < 1976
    );
```

## IN

`IN` can be used to filter results using a list/array.

```SQL
SELECT song, artist
FROM songs
WHERE
    artist IN ('Bob Dylan', 'The Beatles')
ORDER BY release_year;
```

```SQL
SELECT song, artist
FROM songs
WHERE 
    release_year IN (1964, 1966, 1971)
```

## LIKE

`LIKE` can be used to filter string fields using wildcards.

```SQL
SELECT sog
FROM songs
WHERE song LIKE 'a%';
```