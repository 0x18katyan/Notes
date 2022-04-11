# Regular Expressions

## Basic String Manipulation

- Split string according to the newline character `\n` using the `.splitlines()` method.

- Use `.rstrip()` is remove trailing characters such as `\n` and whitespace. Use `.lstrip()` for the same but from the start of the string.

- Use `.strip()` to remove any characters given as argument from a string.

```Python
a = ["977", "9841541231"]

"-".join(a)
## Result: '977-9841541231'
```

The `.join()` method can be used to join strings in a list using a specified character, `-` in the example above, between each string elements while joining.

- Method `.splitlines()` is equal to `.split('\n')`.

### Finding and Replacing

- `.find(substring, start, end)` method can be used for finding substrings in a string. It matches the substring exactly and returns the first index where the substring is seen and `-1` if it doesn't find any matches.

- `.index()` method is similar to `.find()` but it throws a `ValueError()` if the substring isn't found.

- `.count()` method can be used to count the number of occurences of the substring in the string.

- `.replace()` method replaces any substring that matches in the given string with another string.

## String Formatting

- The fstrings style can be used to replace values in the brackets of a string with values from variables.

    ```Python
    var_a = "Variable A"
    var_b = "Variable B"

    print(f"var_a is {var_a} and var_b is {var_b}")
    ```

- `.format()` method can be used to format strings in various ways.

    1. Empty brackets:
        - `print("{} is the first and the second is {}.".format(1, 2))`

    2. With indices:
        - `print("Index 0 is {0} and index 3 is {3} since python starts with 0. Indices 1 and 2 are {1} and {2} respectively for a list of length 4.".format(*[0, 3, 1, 2]))`

    3. Named placeholders:
        - `print("{name} is the name and {game} is the game.".format(name="ashim", game="hide-and-seek"))`

        - Variables can also be passed to the placeholders.

        - Named placeholders can also be combined with other data types.

            ```Python
            some_dict = {"name": "ashim", "game": "hide-and-seek"}
            print("{data[name]} is the name and {data[game]} is the game.".format(data = some_dict))
          ```

    4. Format specifiers are used to denote what data type should be used.

        - `print("{.f} percent of people can understand this.".format(0.9999))`

        - We can also limit the number of digits to print after the decimal point with

            - `print("{.2f} percent of people can understand this.".format(0.9999))`

    5. Datetime objects can also be formatted using string formatting.

        ```Python
        from datetime import datetime
        print("Today's date is {:%Y-%m-%d %H:%M}".format(datetime.now()))
        ```

