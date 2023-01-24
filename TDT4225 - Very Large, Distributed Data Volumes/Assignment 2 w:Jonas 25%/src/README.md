## Configuration of database connection

For running this application, you will have to configure a connection to a MySQL database. We have defined two files which is not uploaded to git, called `.env.local.secret` and `.env.secret`, having the following parameters:

```
HOST=
DATABASE_NAME=
USERNAME=
PASSWORD=
PORT=
```

One of these files have to be created and imported in the file `DbConnector.py`.
