# Dealing with Strings in SQL

## LEN

Calculate the length of strings in a column.

```SQL
SELECT 
   LEN(description) AS description_length 
FROM 
  movies;
```

## LEFT

Select `n` characters starting from left.

```SQL
SELECT 
  LEFT(description, 25) AS first_25_left 
FROM 
  movies;
```

## RIGHT

Select `n` characters starting from right.

```SQL
SELECT 
  RIGHT(description, 25) AS first_25_left 
FROM 
  movies;
```

## CHARINDEX

Find the index for specified substring in rows of a column.

```SQL
SELECT 
  description, 
  CHARINDEX('Love', description) 
FROM 
  movies
WHERE description LIKE '%Love%';
```

The `LIKE` clause is used to denote that `Love` can appear anywhere in the description regardless of the words or characters surrounding it.

## SUBSTRING

`SUBSTRING` helps us to extract strings ranging from `n` to `n_x`.

Let's suppose love is at 13th index of every row then the start for the substring will be start_index + len('love') = 13 + 4 = 17. And the end of the substring will be the length of the description because we want to extract everything after the *Love* substring.

```SQL
SELECT TOP (10)
  description, 
  CHARINDEX('Love', description) AS start_of_string, 
  LEN ('Love') AS length_of_string, 
  SUBSTRING(
    description, 
    17, 
    LEN(description)
  ) AS additional_description 
FROM 
  movies
WHERE description LIKE '%Love%';
```
