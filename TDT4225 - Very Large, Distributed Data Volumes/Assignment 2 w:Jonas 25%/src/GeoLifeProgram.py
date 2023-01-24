from dataclasses import dataclass
from datetime import datetime
import os
import pandas as pd
from DbConnector import DbConnector
from tabulate import tabulate
from database_functions._queries import Queries, Table
from pathlib import Path
from collections import defaultdict
import time
from collections import OrderedDict, defaultdict
from haversine import haversine, Unit
from heapq import nlargest
import sys
import math


@dataclass
class User:
    id: str
    has_labels: bool


@dataclass
class Activity:
    id: int
    user_id: str
    transportation_mode: str
    start_date_time: datetime
    end_date_time: datetime


@dataclass
class TrackPoint:
    activity_id: int
    latitude: float
    longitude: float
    altitude: int
    date_days: float
    date_time: datetime


class Database:
    def __init__(self):
        self.connection = DbConnector()
        self.db_connection = self.connection.db_connection
        self.cursor = self.connection.cursor
        self.queries = Queries()

    def create_tables(self):
        for table in Table.__reversed__():
            self.cursor.execute(self.queries.drop_table(table))
            self.db_connection.commit()
        for table in Table:
            self.cursor.execute(self.queries.create_table(table))
            self.db_connection.commit()

    def fetch_data(self, table, id=None):
        self.cursor.execute(self.queries.SELECT(table, id))
        rows = self.cursor.fetchall()
        # Using tabulate to show the table in a nice way
        print("Data from table %s, tabulated:" % table)
        print(tabulate(rows, headers=self.cursor.column_names))
        return rows

    def drop_table(self, table: Table):
        print(f"Dropping table {table.value} ...")
        self.cursor.execute(self.queries.drop_table(table))

    def show_tables(self):
        self.cursor.execute("SHOW TABLES")
        rows = self.cursor.fetchall()
        print(tabulate(rows, headers=self.cursor.column_names))

    def _fill_with_data(self):
        base_path = Path(__file__).parents[1]

        # Find all labeled users
        labeled_users = set()
        with open(base_path / "./dataset/dataset/labeled_ids.txt", "r") as f:
            for line in f.readlines():
                labeled_users.add(line.strip())

        # Find all paths to users trackings
        data_folder_path = os.path.join(base_path, "./dataset/dataset/Data")
        activity_files = defaultdict(list)
        for plt_path in Path(data_folder_path).rglob("**/*/*.plt"):
            activity_files[str(os.path.normpath(plt_path)).split(os.sep)[-3]].append(plt_path)

        activity_id = 0
        for key, value in sorted(activity_files.items()):
            start = time.time()
            is_labeled = True if key in labeled_users else False
            user = User(key, is_labeled)
            self.cursor.execute(self.queries.INSERT(Table.User, {"id": user.id, "has_labels": user.has_labels}))
            self.db_connection.commit()
            if is_labeled:
                labels_df = pd.read_csv(os.path.join(base_path, f"./dataset/dataset/Data/{key}/labels.txt"),
                                        sep="\t", header=1, names=["Start Time", "End Time", "Transportation Mode"])

            # For paths in users data-folder, each plt_path is a .plt file with TrackPoints
            i = 0
            for plt_path in sorted(value):
                n_trackpoints = (sum(1 for _ in open(plt_path)) - 6) # Ignoring 6 rows of metadata at the top
                if n_trackpoints <= 2500: 
                    trackpoints = OrderedDict()
                    with open(plt_path, "r") as f:
                        for index, row in enumerate(f.readlines()[6:]):
                            split_row = row.strip().split(",")
                            helper = [activity_id] + [split_row[k] for k in range(len(split_row) - 2) if k != 2]
                            date_time_str = ",".join([split_row[5], split_row[6]])
                            helper.append(date_time_str)
                            trackpoints[index] = helper

                    plt_start_datetime = datetime.strptime(trackpoints[0][5], '%Y-%m-%d,%H:%M:%S')
                    plt_end_datetime = datetime.strptime(trackpoints[len(trackpoints) - 1][5], '%Y-%m-%d,%H:%M:%S')

                    # Check if a label exactly matches the beginning and ending times of an activity.
                    # If not, don't label the activity.
                    activity_label = None
                    if is_labeled and i < len(labels_df.index):
                        label_start_datetime = datetime.strptime(labels_df["Start Time"].iloc[i], '%Y/%m/%d %H:%M:%S')
                        while plt_start_datetime > label_start_datetime and i < (len(labels_df.index) - 1):
                            label_start_datetime = datetime.strptime(labels_df["Start Time"].iloc[i + 1], '%Y/%m/%d %H:%M:%S')
                            i += 1

                        label_end_datetime = datetime.strptime(labels_df["End Time"].iloc[i], '%Y/%m/%d %H:%M:%S')
                        if label_start_datetime == plt_start_datetime and label_end_datetime == plt_end_datetime:
                            activity_label = labels_df["Transportation Mode"].iloc[i]

                    self.cursor.execute(self.queries.INSERT(Table.Activity,
                                                            Activity(activity_id,
                                                                     key,
                                                                     activity_label,
                                                                     plt_start_datetime.__str__(),
                                                                     plt_end_datetime.__str__())
                                                            .__dict__
                                                            ))
                    self.db_connection.commit()
                    activity_id += 1

                    values = []
                    for row in trackpoints.values():
                        values.append(row[0])
                        values.append(row[1])
                        values.append(row[2])
                        values.append(row[3])
                        values.append(row[4])
                        values.append(row[5])

                    self.cursor.execute(self.queries.INSERT_BATCH(Table.TrackPoint,
                                                                  ["activity_id",
                                                                   "latitude",
                                                                   "longitude",
                                                                   "altitude",
                                                                   "date_days",
                                                                   "date_time"],
                                                                  values))

                    self.db_connection.commit()
            end = time.time()
            print(f"Time for insertion of {key}: ", end - start)


def tabulate_rows(rows, column_names):
    print(tabulate(rows, headers=column_names))


def task_1(db: Database):
    """
    Find the amount of users, activities and trackpoints.
    """
    db.cursor.execute("SELECT COUNT(DISTINCT(User.id)) AS n_users, "
                      "COUNT(DISTINCT(Activity.id)) AS n_activities, "
                      "COUNT(TrackPoint.id) AS n_trackpoints "
                      "FROM User LEFT JOIN (Activity, TrackPoint) "
                      "ON (User.id = Activity.user_id "
                      "AND Activity.id = TrackPoint.activity_id);")
    print("\nTask 1")
    tabulate_rows(db.cursor.fetchall(), db.cursor.column_names)


def task_2(db: Database):
    """
    Find the average, minimum and maximum number of activities per user.
    """
    # First get the count of all activities per user
    """
        "SELECT COUNT(Activity.id) FROM User "
        "LEFT OUTER JOIN Activity ON (User.id = Activity.user_id) "
        "GROUP BY User.id;"
    """

    # Then use this as a set to answer the max, min, avg task:
    db.cursor.execute(
        "SELECT AVG(x.n_activities) as average, MAX(x.n_activities) as maximum, MIN(x.n_activities) as minimum "
        "FROM (SELECT COUNT(Activity.id) as n_activities FROM User LEFT OUTER JOIN Activity ON (User.id = Activity.user_id) "
        "GROUP BY User.id) as x"
    )
    res_rows = db.cursor.fetchall()
    print("\nTask 2")
    tabulate_rows(res_rows, db.cursor.column_names)

    return res_rows


def task_3(db: Database):
    """
    Find the top 10 users with the highest number of activities.
    """
    db.cursor.execute("SELECT User.id AS uid, "
                      "COUNT(Activity.id) AS n_activities "
                      "FROM User LEFT JOIN (Activity) "
                      "ON (Activity.user_id = User.id) "
                      "GROUP BY User.id "
                      "ORDER BY n_activities "
                      "DESC LIMIT 10;")
    print("\nTask 3")
    tabulate_rows(db.cursor.fetchall(), db.cursor.column_names)


def task_4(db: Database):
    """
    Find the number of users that have started the activity in one day and ended the activity the next day.
    """

    # Query giving a clean table overview of all activities satisfying the task requirements
    "SELECT user.id, activity.id, activity.start_date_time as start, activity.end_date_time as end "
    "FROM user JOIN activity ON(user.id = activity.user_id) "
    "WHERE (SELECT DATEDIFF(activity.end_date_time, activity.start_date_time)) > 0;"

    # Query returning only count of all these activities:
    db.cursor.execute(
        "SELECT COUNT(a) as 'Users with multi-day activities' FROM"
        "(SELECT DISTINCT user.id as a FROM "
        "user JOIN activity ON(user.id = activity.user_id) "
        "WHERE DATEDIFF(activity.end_date_time, activity.start_date_time) > 0 "
        "AND DATEDIFF(activity.end_date_time, activity.start_date_time) < 2) as x;"
    )
    print("\nTask 4")
    tabulate_rows(db.cursor.fetchall(), db.cursor.column_names)


def task_5(db: Database):
    """
    Find activities that are registered multiple times. You should find the query even if you get zero results.
    """
    db.cursor.execute("SELECT a1.user_id AS user, a1.id AS id_1, a2.id AS id_2 FROM Activity AS a1 JOIN Activity AS a2 ON (a1.start_date_time = a2.start_date_time AND a1.end_date_time = a2.end_date_time) WHERE (a1.user_id = a2.user_id AND a1.id != a2.id);")
    print("\nTask 5")
    tabulate_rows(db.cursor.fetchall(), db.cursor.column_names)


def task_7(db: Database):
    """
    Find all users that have never taken a taxi.
    """
    db.cursor.execute(
        "SELECT * FROM User WHERE (User.id NOT IN (SELECT User.id FROM User LEFT JOIN (Activity) "
        "ON (User.id = Activity.user_id) WHERE (transportation_mode = 'taxi')));")
    print("\nTask 7")
    res = db.cursor.fetchall()
    tabulate_rows(res, db.cursor.column_names)
    print("Amount of users that have never taken a taxi: ", len(res))
    # For checking how many users have taken taxi, run
    # SELECT COUNT(DISTINCT(User.id)) FROM User LEFT JOIN (Activity) ON (User.id = Activity.user_id) where (transportation_mode = "taxi")
    # 182-(this number) should be n_rows, which it is


def task_8(db: Database):
    """
    Find all types of transportation modes and count how many distinct users that have used the different transportation modes.
    Do not count the rows where the transportation mode is null.
    """
    db.cursor.execute(
        "SELECT transportation_mode, COUNT(DISTINCT(user_id)) AS unique_uids "
        "FROM Activity WHERE transportation_mode IS NOT NULL "
        "GROUP BY transportation_mode;")
    print("\nTask 8")
    tabulate_rows(db.cursor.fetchall(), db.cursor.column_names)


def task_9(db: Database):
    """
    a) Find the year and month with the most activities.
    b) Which user had the most activities this year and month, and how many recorded hours do they have?
        Do they have more hours recorded than the user with the second most activities?
    """

    # Mode month not dependent on year
    """
    db.cursor.execute("SELECT DISTINCT @mode_month := MONTHNAME(start_date_time) as month_name, "
                      "Count(Month(start_date_time)) as month_name_occurrences "
                      "FROM Activity GROUP BY month_name "
                      "ORDER BY month_name_occurrences "
                      "DESC limit 1;")
    db.cursor.fetchall()
    db.cursor.execute("SELECT DISTINCT @mode_year := YEAR(start_date_time) as year, "
                      "Count(YEAR(start_date_time)) as year_occurrences "
                      "FROM Activity "
                      "GROUP BY year "
                      "ORDER BY year_occurrences "
                      "DESC limit 1;")
    db.cursor.fetchall()
    db.cursor.execute("SELECT @mode_month as 'Month with most activities', "
                      "@mode_year as 'Year with most activities';")
    """
    # Makes more sense to use the mode_month dependent on mode_year in task 9b, so we assume this.
    db.cursor.execute("SELECT DISTINCT @mode_year := YEAR(start_date_time) as year, "
                      "@mode_month := MONTHNAME(start_date_time) as month_name, "
                      "Count(YEAR(start_date_time)) as year_occurrences "
                      "FROM Activity "
                      "GROUP BY year, month_name "
                      "ORDER BY year_occurrences;")
    db.cursor.fetchall()
    db.cursor.execute("SELECT @mode_year as 'Year with most activities', "
                      "@mode_month as 'Month with most activities in this year';")

    print("\nTask 9a")
    tabulate_rows(db.cursor.fetchall(), db.cursor.column_names)

    # MySQL hours takes floor of each timediff
    """
    db.cursor.execute(
        "SELECT DISTINCT User.id as uid, "
        "Count(activity.id) as n_activities, "
        "SUM(HOUR(TIMEDIFF(activity.end_date_time, activity.start_date_time))) as hours "
        "FROM USER LEFT JOIN (Activity) ON (User.id = Activity.user_id) "
        "WHERE (MONTHNAME(activity.start_date_time) = @mode_month "
        "AND YEAR(activity.start_date_time) = @mode_year) "
        "GROUP BY User.id "
        "ORDER BY n_activities "
        "DESC LIMIT 2;"
    )
    """
    # Count all elapsed time in seconds and divide by 3600 for correct amount of hours
    db.cursor.execute(
        "SELECT DISTINCT User.id as uid, "
        "Count(activity.id) as n_activities, "
        "SUM(TIMESTAMPDIFF(SECOND, activity.start_date_time, activity.end_date_time)/3600) as hours "
        "FROM USER LEFT JOIN (Activity) ON (User.id = Activity.user_id) "
        "WHERE (MONTHNAME(activity.start_date_time) = @mode_month "
        "AND YEAR(activity.start_date_time) = @mode_year) "
        "GROUP BY User.id "
        "ORDER BY n_activities "
        "DESC LIMIT 2;"
    )
    print("\nTask 9b")
    tabulate_rows(db.cursor.fetchall(), db.cursor.column_names)


def task_10(db: Database):
    """
    Find the total distance (in km) walked in 2008, by user with id = 112.
    """
    db.cursor.execute(
        "SELECT * FROM trackpoint "
        "JOIN activity ON(trackpoint.activity_id = activity.id) "
        "WHERE activity.user_id = 112 "
        "AND YEAR(trackpoint.date_time) = 2008 "
        "AND activity.transportation_mode = 'walk' "
        "ORDER BY activity.id, trackpoint.date_time;"
    )
    res_rows = db.cursor.fetchall()

    dist = 0
    prev_tp = res_rows[0]  # (res_rows[0][2], res_rows[0][3])
    for tp in res_rows[1:]:
        if tp[1] != prev_tp[1]:
            continue
        prev_coords = (prev_tp[2], prev_tp[3])
        new_coords = (tp[2], tp[3])
        dist += haversine(prev_coords, new_coords)
        prev_tp = tp
    print("\nTask 10")
    print(f"User_id: 112\nDistance in 2008: {dist}")

    return dist


def task_11(db: Database):
    """
    Find the top 20 users who have gained the most altitude meters.
        ○ Output should be a table with (id, total meters gained per user).
        ○ Remember that some altitude - values are invalid
        ○ Tip: ∑(tp_{n}.altitude − tp_{n−1}.altitude), tp_{n}.altitude > tp_{n−1}.altitude
    """
    def gain_alt(alt1, alt2):
        return alt2 - alt1 if alt1 < alt2 else 0

    # Get amount of users in total
    db.cursor.execute("SELECT COUNT(*) FROM user;")
    num_users = int(db.cursor.fetchall()[0][0])

    db.cursor.execute(
        "SELECT activity.user_id, trackpoint.altitude, trackpoint.date_time, trackpoint.activity_id FROM trackpoint "
        "JOIN activity ON(trackpoint.activity_id = activity.id) "
        "WHERE trackpoint.altitude <> -777 "
        "ORDER BY trackpoint.activity_id, trackpoint.date_time ASC;"
    )
    res_rows = db.cursor.fetchall()

    alt_per_user = [0] * num_users

    prev_tp = res_rows[0]
    for tp in res_rows[1:]:
        # Ensure same acitivity and user
        uid = int(tp[0])
        if tp[3] != prev_tp[3] or uid != int(prev_tp[0]):
            prev_tp = tp
            continue
        
        alt_gain = gain_alt(prev_tp[1], tp[1])
        alt_per_user[uid] += alt_gain
        prev_tp = tp

    n_highest_gainers = nlargest(20, enumerate(alt_per_user), key=lambda x: x[1])

    print("\nTask 11")
    tabulate_rows(n_highest_gainers, ["uid", "m gained"])

    return n_highest_gainers


def task_12(db: Database):
    """
    Find all users that have invalid activities, and the number of invalid activities per user
        ○ An invalid activity is defined as an activity with consecutive trackpoints
            where the timestamps deviate with at least 5 minutes.
    """

    db.cursor.execute("SELECT COUNT(*) FROM user;")
    num_users = int(db.cursor.fetchall()[0][0])

    db.cursor.execute("SELECT activity.user_id, activity.id, trackpoint.date_time "
                      "FROM trackpoint JOIN activity ON(trackpoint.activity_id = activity.id) "
                      "ORDER BY activity.id, trackpoint.date_time ASC;")
    res_rows = db.cursor.fetchall()

    invalid_per_user = [0] * num_users

    prev_tp = res_rows[0]
    for tp in res_rows[1:]:
        # Ensure same acitivity and user
        if tp[1] != prev_tp[1] or tp[0] != prev_tp[0]:
            prev_tp = tp
            continue
        
        time_diff = (tp[2] - prev_tp[2]).total_seconds()
        if time_diff >= 300:
            invalid_per_user[int(tp[0])] += 1
        prev_tp = tp

    print("\nTask 12")
    print(num_users)
    tabulate_rows([(i, invalid_per_user[i]) for i in range(num_users)], ["uid", "Invalid activities"])

    return invalid_per_user

"""
From this point on to main is only testing for task 6.
"""
class Day:
    def __init__(self):
        self.parts = [[]] * 8


day_interval = 3
day_parts = math.floor(24 / day_interval)


def get_day_part(d):
    # Method used to separate trackpoints into finer grained parts of a day.
    # Also adds some overlap to make trackpoints that are close enough in time, but not on the same day be compared.
    minutes = d.minute
    hours = d.hour
    p1 = math.floor(hours / day_interval)
    p2 = -1
    if p1 == day_parts - 1:
        if minutes == 59:
            p2 = 0
    elif p1 == 0:
        if minutes == 0:
            p2 = 7
    return (p1, p2)


class tp_node:
    def __init__(self, before, after, data):
        self.before = before
        self.after = after
        self.tp = data


def check_time_window(d1, d2):
    return True if abs(d2 - d1).total_seconds() <= 60 else False


def check_distance(cord1, cord2):
    dist = haversine(cord1, cord2)
    closeby = True if dist <= 100 else False
    return (closeby, dist)


def task_6_v2(db: Database):
    """
        This way of doing task 6 should be alot faster than the naive approach used in task_6_v1.
        We are however constrained by memory stopping program execution.
        A possible improvement here could be the implementation of batch processing of the data,
        only fetching data from the db after n amount of rows have been processed.
    """


    # Get all trackpoints ordered by date, earliest first.
    db.cursor.execute(
        "SELECT trackpoint.id, latitude, longitude, date_time, user_id FROM trackpoint "
        "JOIN activity ON(trackpoint.activity_id = activity.id) "
        "ORDER BY date_time ASC"
    )
    rows = db.cursor.fetchall()

    hit_list = []

    print("Start insert")

    day_dict = defaultdict(Day) # Dict to separate trackpoints into days, contains Day objects containing arrays for further separation
    for row in rows:
        date_time = row[3]
        day_part1, day_part2 = get_day_part(date_time)
        day_dict[date_time.date()].parts[day_part1].append(row)
        if day_part2 != -1:
            day_dict[date_time.date()].parts[day_part2].append(row)

    print("Done inserting\nLength of day_dict: ", len(day_dict.keys()))

    day_x = 0
    for key, day in day_dict.items():
        #print("Date: ", key, "\nDay ", day_x)
        n_tp = 0
        hits = 0
        for day_part in day.parts:

            # Making a linked list for use in comparing trackpoints.
            # Elements not in the linked list do not need to be compared with the elements not already in the list
            first_tp = tp_node(None, None, day_part[0])
            last_tp = first_tp
            for tp in day_part[1:]:
                new_tp = tp_node(last_tp, None, tp)
                last_tp.after = new_tp
                last_tp = new_tp

                check_tp = first_tp
                while check_tp.after != last_tp:
                    if check_time_window(new_tp.tp[3], check_tp.tp[3]):
                        if check_tp.tp[4] != new_tp.tp[4]:
                            cord1 = (new_tp.tp[1], new_tp.tp[2])
                            cord2 = (check_tp.tp[1], check_tp.tp[2])
                            close, dist = check_distance(cord1, cord2)
                            if close:
                                hits += 1
                                hit_list.append((new_tp.tp[4], check_tp.tp[4], new_tp.tp[3], dist))
                    else:
                        check_tp.after.before = None
                        if check_tp == first_tp:
                            first_tp = check_tp.after
                    check_tp = check_tp.after
                n_tp += 1
        print("N_tp: ", n_tp, "\nHits: ", hits)
        day_x += 1

    print("DONE\nTask 6\n")
    print(hit_list)
    return hit_list


def task_6_v1(db: Database):
    """
    Inefficient way to do this task. Won't complete before a few days or more have passed...
    This is probably due to the cartesian product producing a ridiculous amount of rows to be processed.
    """

    db.cursor.execute(
        "SELECT t1.id, t1.latitude, t1.longitude, t1.altitude, t1.date_time, t1.user_id, t2.id, "
        "t2.latitude, t2.longitude, t2.altitude, t2.date_time, t2.user_id FROM "
        "(SELECT trackpoint.*, activity.user_id FROM trackpoint JOIN activity ON(trackpoint.activity_id = activity.id)) as t1 "
        "JOIN (SELECT trackpoint.*, activity.user_id FROM trackpoint JOIN activity ON(trackpoint.activity_id = activity.id)) as t2 "
        "WHERE t1.id <> t2.id AND t1.activity_id <> t2.activity_id AND t1.user_id <> t2.user_id "
        "AND ABS(UNIX_TIMESTAMP(t1.date_time) - UNIX_TIMESTAMP(t2.date_time)) < 61;"
    )
    rows = db.cursor.fetchall()

    encounters = []
    for row in rows:
        if haversine((row[1], row[2]), (row[7], row[8])) <= 100:
            encounters.append((row[5], row[11], row[4]))

    tabulate_rows(encounters, ["uid_1", "uid_2", "Approx. time"])


if __name__ == '__main__':
    db = None

    try:
        db = Database()
        # db.create_tables()
        # db._fill_with_data()
        # db.fetch_data(table=Table.Activity)

        stdout_temp = sys.stdout
        #sys.stdout = open("output.txt", "w+")

        # Tasks
        task_1(db)
        # task_2(db)
        # task_3(db)
        # task_4(db)
        # task_5(db)
        # task_6_test(db)
        #task_7(db)
        #task_8(db)
        #task_9(db)
        #task_10(db)
        #task_11(db)
        task_12(db)

        # db.drop_table(table=Table.User)
        # Check that the table is dropped
        # db.show_tables()
        #sys.stdout = sys.__stdout__
        print("Results written to file: 'output.txt' :)")

    finally:
        if db:
            db.connection.close_connection()
            sys.stdout.close()
