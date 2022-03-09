# Database Access Control

## Database roles

- Manage database access permissions.
- Contains information such as:
    1. Role's privileges:
        - Can you login?
        - Can you create databases?
        - Can you write to tables?

    2. Interact with the client authentication system:
        - Password.
- Roles can be assigned to one or more users.
- Roles are global across a database cluster installation.

A role can be a user role or a group role. 

### Create role

#### Create empty role

```SQL
CREATE ROLE data_analyst;
```

#### Create roles with attributes

##### Create with password and password expiry date

```SQL
CREATE ROLE intern WITH PASSWORD 'PasswordForIntern' VALID UNTIL '2022-03-03';
```

##### Create with CREATEDB permission

```SQL
CREATE ROLE admin CREATEDB;
```

##### Grant admin with CREATEROLE prvilege

```SQL
ALTER ROLE admin CREATEROLE;
```

### Grant Role

```SQL
GRANT role_name to username;
```

### Revoke Role

```SQL
REVOKE role_name FROM username
```

### Give role a password

```SQL
ALTER ROLE role_name WITH PASSWORD 's3cur3p@ssw0rd';
```
