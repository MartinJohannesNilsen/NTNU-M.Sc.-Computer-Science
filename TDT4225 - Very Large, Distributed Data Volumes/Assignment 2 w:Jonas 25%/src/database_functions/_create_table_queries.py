CREATE_TABLE_QUERIES = {}
CREATE_TABLE_QUERIES["User"] = (
    "CREATE TABLE User ("
    "`id` VARCHAR(5) NOT NULL,"
    "`has_labels` BOOLEAN,"
    "PRIMARY KEY (`id`)"
    ");")

CREATE_TABLE_QUERIES["Activity"] = (
    "CREATE TABLE Activity("
    "`id` INTEGER NOT NULL,"
    "`user_id` VARCHAR(5) NOT NULL,"
    "`transportation_mode` VARCHAR(50),"
    "`start_date_time` DATETIME NOT NULL,"
    "`end_date_time` DATETIME NOT NULL,"
    "PRIMARY KEY (`id`),"
    "FOREIGN KEY (`user_id`)"
    "REFERENCES `User` (`id`)"
    "ON DELETE CASCADE"
    ");")

CREATE_TABLE_QUERIES["TrackPoint"] = (
    "CREATE TABLE TrackPoint("
    "`id` INTEGER NOT NULL AUTO_INCREMENT,"
    "`activity_id` INTEGER NOT NULL,"
    "`latitude` DOUBLE NOT NULL,"
    "`longitude` DOUBLE NOT NULL,"
    "`altitude` INTEGER NOT NULL,"
    "`date_days` DOUBLE NOT NULL,"
    "`date_time` DATETIME NOT NULL,"
    "PRIMARY KEY (`id`),"
    "FOREIGN KEY (`activity_id`)"
    "REFERENCES `Activity` (`id`)"
    "ON DELETE CASCADE"
    ");")
