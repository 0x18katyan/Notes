# Notes

- Use `client.list_database_names()` for listing databases in a MongoDB instance.
- Use `db_name.list_collection_names()` for listing collections in a Database.

- `db_name.collection_name.find_one()` can be used to fetch a single document from the database.
    1. This is can used in conjunction with a filter argument to find specific documents.
    2. This can also be used to learn about the inherent structure of the documents stored in a collection but beware not all the documents share the same structure.

## Filter

### Exact Matches

- A filter can be as simple as `{'somefield': 'somevalue'}`.

Nested Filter

```Python
filter_doc = {
    'someField1': 'someValue1',
    'someField2': 'someValue2',
    'someField3': 'someValue3',
}

## this counts the number of documents that satisfy the above filter
db.some_collection.count_documents(filter_doc) 
```

### Exists in List

```Python

filter_doc = {
    'someField': {
        '$in' : ['someValue1', 'someValue2'] 
        ## the $in is used to search for documents that have either someValue1 or someValue2 for their someField
        ## Also note that the someField has a document for value instead of a string for exact matches
    }
}

db.some_collection.count_documents(filter_doc)
```

### Not Equal

```Python
filter_doc = {
    'someField': {
        '$ne' : 'someValue'
    }
}

db.some_collection.count_documents(filter_doc)
```

### Comparison

- `gt`: greater than
- `lt`: less than
- `gte`: greater than or equal to
- `lte`: less than or equal to

```Python
filter_doc = {
    'someField': {
        '$gt': 'someValue1',
        '$lte': 'someValue2'
    }
}
db.some_collection.count_documents(filter_doc)
```

- Comparison operations between strings is permitted but MongoDB compares the strings, if no numbers are present, in alphabetical order. Ex: a,b,c,e etc.

- Weird behavior: the value being string or integer while comparing affects the results. Need to know more about it.

### Exists

Exists is used for checking if a document has a field or not.

```Python
filter_doc = {
    'someField': {
        '$exists': True
    }
} 
```

### Dot Notation

Nested Fields can be accessed using the dot notation.

```Python
# someRootFieldStructure = {
    # 'someField': { 
        # 'someNestedField1': 'someVal2', 'someNestedField2': 'someVal2'
    # } 
# }

filter_doc = {'someField.someNestedField1' :{'$ne': 'someField.someNestedField2'}}
```

Check if document has alteast 4 fields, lists start with 0.

```Python
## someDoc = {'someField': [1,2,3,4,5]}

filter_doc = {"someField.3": {'$exists': True}}
```

```Python
countries = set(db.laureates.distinct('diedCountry')) - set(db.laureates.distinct('bornCountry'))
```


```Python
# The number of distinct countries of laureate affiliation for prizes
count = len(db.laureates.distinct('prizes.affiliations.country'))
print(count)
```

```Python
filter_doc = {
    "prizes": {
        "$elemMatch": {
            "category": "physics",
            "share": "1",
            "year": {"$lt": "1945"}
        }
    }
}

filter_doc = {
    "prizes": {
        "$elemMatch": {
            "category": "physics",
            "share": "1",
            "year": {"$lt": "1945"}
        }
    }
}

```

- `$nin` for not in.

Regex

```Python 
criteria = {"bornCountry": Regex("now" + " Germany\)" + "$")}
```