# Notes on Union

`UNION` can be used to combine the results of two queries if they have identical columns and column data types. Union discards duplicate results.

```SQL
SELECT
    song,
    artist,
FROM songs
WHERE artist IN ('Bob Dylan', 'The Beatles')
UNION
SELECT
    song,
    artist
FROM songs
WHERE artist IN ('Pink Floyd', 'Bob Dylan', 'Led Zeppelin');
```

`UNION ALL` can be used to return duplicate results.

```SQL
SELECT
    song,
    artist,
FROM songs
WHERE artist IN ('Bob Dylan', 'The Beatles')
UNION ALL
SELECT
    song,
    artist
FROM songs
WHERE artist IN ('Pink Floyd', 'Bob Dylan', 'Led Zeppelin');
