# Introduction

Scala is a general-purpose programming language. Scala is related to Java. Scala source code compiles to Java bytecode and runs on Java Virtual Machines.

Scala is a SCAlable LAnguage.

It is flexible. New types, collections and control constructs.

## Why use Scala?

Scala fuses Object Oriented Programming and Functional Programming.

Object Oriented Programming:

1. Everything is an object in Scala.
2. Every operation is a method call.

Functional Programming:

1. Functions are first-class values which means they can be passed as variables too.

2. Program should generate new output by performing operations on input but should not change the input in any way.

Scala is a high-level language. Scala is statically typed.

Functions and methods are *not the same* in Scala.

## Variables in Scala

### Variable Types

### Val

Val is Immutable.

```Scala
val fourHearts: Int = 4
```

#### Var

Var is Mutable.

```Scala
var fourHearts: Int = 4
```

### Value Types in Scala

- Double
- Float
- Long
- Int
- Short
- Byte
- Char
- Boolean
- Unit

`Double` is the default floating point value type for scala.

`Int` is 32-bit.

`String` is a sequence of `Char`.

Scala value types have equivalent Java types.

Scala can do type inference.

#### Declaring a Variable

##### With Type Definition

```Scala
val someString: String = "This is a string in Scala."
```

Notice that the variable type is `val` so the string is immutable or cannot be reassigned.

```Scala
var ace: Int = 1
ace = 11
```

##### Type Inference

```Scala
var ace = 1
```

### Why use Immutability?

#### Pros

- Less prone to errors.
- Less tracking for changes in variables.
- Have to write less test scenarios because of immutability.

#### Cons

- Uses more memory due to copying operations needed when changing the values in the immutable variable.

## Basic Syntax

### Print

```Scala
println(1 + 2)
```

### Comment

```Scala
// this is a single-line comment.
```

## Scala Scripts

- Sequence of instructions in a file which is executed sequentially.

- Useful for smaller projects.

- Scala Interpreter executes the script by wrapping it up in a template and then compiling it and executing the resultant compiled program.

## Scala applications

- Compiled explicitly then run explicitly.
- Consists of various source files that can be compiled individually.
- Is useful for larger programs.
- No lag time since its precompiled.

### Compiling Scala

```Bash
// Compile an App
scalac App.scala

// Run the compiled App
scala App
```

`sbt` is the most commonly used Scala compiler.

[almond](https://almond.sh) is a scala kernel for Jupyter Notebooks.

## Functions in Scala

- Functions are invoked with a list of arguments to produce a result.

- Functions consist of parameter list, body and result type.

- Functions are fist-class values.

- All functions produce results.

- All results have values.

- All values have value types.

- Functions can be passed as arguments to other functions.

## Examples

Example 1

```Scala
def func_name( arg1: Int, arg2: Int) : Int = {
    sum = arg1 + arg2
}
```

**Note: There is no way to conditionally return in Scala. The last statement of the function is regarded as the return value. The usage of the return keyword is highly frowned upon and goes against the design principal of Scala.**

Example 2

```Scala
def maxNum(numA: Int, numB: Int): Int = {
    if (numA > numB) numA
    else numB
}
```

## Collections

Collections can be either mutable or immutable.

### Array

- Mutable sequence of objects that share the same type.

Dynamically defined arrays:

```Scala
val arrayInt = Array(1, 2, 3, 4)
var arrayStr = Array("one", "two", "three")
```

Statically defined arrays:

```Scala
// array of String type defined with a length of 3
val array = new Array[String](3)

array(0) = "one"
array(1) = "two"
array(2) = "three"
```

```Scala
var numArray: Array[Int] = new Array[Int](3)
```

Arrays can only be updated with the same type.

**Scala nudges users to immutability.**

`Any` type can be used to form an array with mixed types.

```Scala
var anyArray Array[Any] = new Array[Any](3)
```

Iterating over elements in array:

```Scala
anyArray.foreach(println)
```

### Lists

`Lists` are immutable in Scala. Same as Array but just immutable.

Prepend to list using:

```Scala
val numList = List(10, 15, 20 , 25, 30)
val newNumList = 5 :: numList
```

Prepend List to List

```Scala
val thirdList = firstList ::: secondList
```

## Conditionals

```Scala
val someVal = 10

if (someVal < 10) {
    println("Value greater or equal to 10.")
}
```

Style 1:

```Scala
val someVal = 10

if (someVal < 10) {
    println("Value greater than or equal to 10.")
} else println("Value less than 10.")
```

Style 2:

```Scala
if (someVal > 10)
    println("Value greater than or equal to 10.")
else
    println("Value less than 10.")
```

Style 3:

```Scala
if (someVal > 10) println("Value greater than or equal to 10.")
else println("Value less than 10.")
```

Style 4:

```Scala
if (someVal > 10) println("Value greater than or equal to 10." else println("Value less than 10.")
```

## Operators

### Relational

Greater than : `>`

Less than : `<`

Greater than or equal to : `>=`

Less than or equal to : `<=`

Equal to : `==`

Not equal to : `!=`

### Logical

And : `&&`

Or : `||`

Not : `!`

## Iterators

### While Loop

```Scala
var i = 0
val numRepeat = 3

while (i < numRepeat) {
    println(i)
    i += 1
    // i = i + 1
}
```

```Scala

val list1 = List(1, 2, 3)
println(list1.length) // prints the length of the list

```

**Side effect is code modifying the variable outside of its local scope.**

