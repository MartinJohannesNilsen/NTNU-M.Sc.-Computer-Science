from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import pandas as pd
from pymongo.database import Database
from DbConnector import DbConnector
from pathlib import Path
from collections import OrderedDict, defaultdict
from pprint import pprint
from statistics import mean
from tabulate import tabulate
from haversine import haversine
from heapq import nlargest
import sys


@dataclass
class User:
    _id: str
    has_labels: bool
    activities: list


@dataclass
class Activity:
    _id: int
    user_id: str
    transportation_mode: str
    start_date_time: datetime
    end_date_time: datetime
    # trackpoints: list


@dataclass
class TrackPoint:
    _id: int
    latitude: float
    longitude: float
    altitude: int
    date_days: float
    date_time: datetime
    activity_id: int


class Database_instance:
    def __init__(self):
        self.connection = DbConnector()
        self.client = self.connection.client
        self.db = self.connection.db

    def create_collection(self, collection_name):
        collection = self.db.create_collection(collection_name)
        print('Created collection: ', collection)

    def insert_documents(self, collection_name: str, documents: list):
        """Insert array of documents

        Args:
            collection_name (str): Name of collection
            documents (list): List of documents on json/dict format
        """
        collection = self.db[collection_name]
        collection.insert_many(documents)

    def fetch_documents(self, collection_name: str, criteria={}):
        """Fetch all documents or by criteria

        Args:
            collection_name (str): Name of collection
            criteria (dict, optional): Criteria for find. Defaults to {}.
        """
        collection = self.db[collection_name]
        documents = collection.find(criteria)
        for doc in documents:
            pprint(doc)

    def drop_collection(self, collection_name: str):
        collection = self.db[collection_name]
        collection.drop()

    def show_collections(self):
        collections = self.client[self.db.name].list_collection_names()
        print(collections)

    def _fill_with_data(self):
        db_users: list(User) = []
        db_activities: list(Activity) = []
        db_trackpoints: list(TrackPoint) = []
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
        trackpoint_id = 0
        for key, value in sorted(activity_files.items()):
            is_labeled = True if key in labeled_users else False
            user_activities = []
            if is_labeled:
                labels_df = pd.read_csv(os.path.join(base_path, f"./dataset/dataset/Data/{key}/labels.txt"),
                                        sep="\t", header=1, names=["Start Time", "End Time", "Transportation Mode"])

            # For paths in users data-folder, each plt_path is a .plt file with TrackPoints
            i = 0
            for plt_path in sorted(value):
                n_trackpoints = (sum(1 for _ in open(plt_path)) - 6)  # Ignoring 6 rows of metadata at the top
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

                    db_activities.append(Activity(activity_id, key, activity_label, plt_start_datetime, plt_end_datetime).__dict__)
                    user_activities.append(activity_id)

                    for row in trackpoints.values():
                        db_trackpoints.append(TrackPoint(_id=trackpoint_id,
                                                         activity_id=activity_id,
                                                         latitude=row[1],
                                                         longitude=row[2],
                                                         altitude=row[3],
                                                         date_days=row[4],
                                                         date_time=datetime.strptime(row[5], '%Y-%m-%d,%H:%M:%S'),
                                                         ).__dict__)
                        trackpoint_id += 1
                    activity_id += 1
            db_users.append(User(key, is_labeled, user_activities).__dict__)
        self.insert_documents(collection_name="User", documents=db_users)
        self.insert_documents(collection_name="Activity", documents=db_activities)
        self.insert_documents(collection_name="TrackPoint", documents=db_trackpoints)


def tabulate_rows(rows, column_names):
    print(tabulate(rows, headers=column_names, floatfmt=".4f"))


def task_1(db: Database):
    """
    Find the amount of users, db_activities and db_trackpoints.
    """
    # Number of users
    n_users = db.User.count_documents({})

    # Number of activities
    n_activities = db.Activity.count_documents({})

    # Number of trackpoints
    n_trackpoints = db.TrackPoint.count_documents({})

    print("\nTask 1")
    tabulate_rows([[n_users, n_activities, n_trackpoints]], ["n_users", "n_activities", "n_trackpoints"])


def task_2(db: Database):
    """
    Find the average, minimum and maximum number of activities per user.
    """
    # This need the User to have a list of activity_ids, which may be unnecessary
    users: list = db.User.find()
    n_activities = []
    for user in users:
        n_activities.append(len(user["activities"]))
    average = mean(n_activities)
    minimum = min(n_activities)
    maximum = max(n_activities)

    print("\nTask 2")
    tabulate_rows([[average, maximum, minimum]], ["average", "maximum", "minimum"])


def task_2_w_pipeline(db: Database):
    """
    Find the average, minimum and maximum number of activities per user.
    """
    # Should outer left join for including the users without activities
    pipeline = [
        {"$lookup": {
            "from": "Activity",
            "localField": "_id",
            "foreignField": "user_id",
            "as": "activities"},
         },
        {"$group": {"_id": "$_id", "count": {"$sum": {"$size": "$activities"}}}},
        {"$sort": {"_id": 1}}
    ]

    res = db.User.aggregate(pipeline)
    n_activities = []
    for row in res:
        n_activities.append(list(row.values())[1])
    average = mean(n_activities)
    minimum = min(n_activities)
    maximum = max(n_activities)

    print("\nTask 2")
    tabulate_rows([[average, maximum, minimum]], ["average", "maximum", "minimum"])


def task_3(db: Database):
    """
    Find the top 10 users with the highest number of activities.
    """
    pipeline = [
        {"$group": {"_id": "$user_id", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]

    count_objects = db.Activity.aggregate(pipeline)
    print("\nTask 3")
    tabulate_rows([(obj["_id"], obj["count"]) for obj in count_objects], ["Id", "Count"])


def task_4(db: Database):
    """
    Find the number of users that have started the activity in one day and ended the activity the next day.
    """
    activities = db.Activity.find({})
    user_ids = set()
    for activity in activities:
        if (activity["end_date_time"].date() - activity["start_date_time"].date()) == timedelta(days=1):
            user_ids.add(activity["user_id"])
    print("\nTask 4")
    tabulate_rows([[len(user_ids)]], ["users with multi-day activities"])


def task_5(db: Database):
    """
    Find activities that are registered multiple times. You should find the query even if you get zero results.
    """
    # This query returns no results, so there are no activities fulfilling the requirements.
    pipeline = [
        {
            "$group": {
                "_id": {
                    "uid": "$user_id",
                    "start": "$start_date_time"
                },
                "count": {"$sum": 1}
            },
        },
        {"$match": {"count": {"$gt": 1}}},  # Find entries that have count greater than 1
    ]

    res = db.Activity.aggregate(pipeline)
    print("\nTask 5")
    tabulate_rows([row for row in res], ["activities registred multiple times"])


def task_6(db: Database):
    """
    An infected person has been at position (lat, lon) (39.97548, 116.33031) at time ‘2008-08-24 15:38:00’.
    Find the user_id(s) which have been close to this person in time and space (pandemic tracking).
    Close is defined as the same minute (60 seconds) and space (100 meters).
    """
    infected_coords = (39.97548, 116.33031)
    infected_time = datetime.strptime("2008-08-24 15:38:00", "%Y-%m-%d %H:%M:%S")

    activities_matching = set()
    trackpoints = db.TrackPoint.find({})

    def check_time_window(d1, d2):
        return True if abs(d2 - d1).total_seconds() <= 60 else False

    def check_distance(cord1, cord2):
        dist = haversine(cord1, cord2)
        closeby = True if dist <= 100 else False
        return (closeby, dist)

    for tp in trackpoints:
        if check_time_window(tp["date_time"], infected_time):
            coords = (float(tp["latitude"]), float(tp["longitude"]))
            close, _ = check_distance(coords, infected_coords)
            if close:
                activities_matching.add(tp["activity_id"])

    activities = db.Activity.find({"_id": {"$in": list(activities_matching)}})
    users_in_contact = [a["user_id"] for a in activities]
    print("\nTask 6")
    print(f"Users in contact: {users_in_contact}")


def task_7(db: Database):
    """
    Find all users that have never taken a taxi.
    """
    taxi_activities = db.Activity.distinct("user_id", {"transportation_mode": "taxi"})
    users_not_taxi = list(db.User.find({"_id": {"$nin": taxi_activities}}))

    print("\nTask 7")
    print("users: ", len(users_not_taxi))
    tabulate_rows(sorted([(row["_id"], row["has_labels"]) for row in users_not_taxi],
                         key=lambda x: -int(x[1])), ["Users that have never taken a taxi", "labeled"])


def task_8(db: Database):
    """
    Find all types of transportation modes and count how many distinct users that have used the different transportation modes.
    Do not count the rows where the transportation mode is null.
    """
    activities = db.Activity.find({})
    transportation_res = defaultdict(set)
    for activity in activities:
        transportation_res[activity["transportation_mode"]].add(activity["user_id"])
    del transportation_res[None]
    print("\nTask 8")
    tabulate_rows([(mode, len(count)) for mode, count in sorted(transportation_res.items())], ["transportation_mode", "Count"])


def task_9(db: Database):
    """
    a) Find the year and month with the most activities.
    b) Which user had the most activities this year and month, and how many recorded hours do they have?
        Do they have more hours recorded than the user with the second most activities?
    """
    # a
    pipeline = [
        {
            "$group": {
                "_id": {
                    "y": {"$year": "$start_date_time"},
                    "m": {"$month": "$start_date_time"}
                },
                "count": {"$sum": 1},
            },
        },
        {"$sort": {"count": -1}},
        {"$limit": 1}
    ]

    res = db.Activity.aggregate(pipeline)
    print("\nTask 9a")
    tabulate_rows([[row["_id"]["y"], row["_id"]["m"]] for row in res], ["Year with most activities", "Month with most activities in this year"])

    # b
    p = [
        {
            "$group": {
                "_id": {
                    "uid": "$user_id",
                    "y": {"$year": "$start_date_time"},
                    "m": {"$month": "$start_date_time"}
                },
                "count": {"$sum": 1},
                "timeSpent": {
                    "$sum": {
                        "$dateDiff": {
                            "endDate": "$end_date_time",
                            "startDate": "$start_date_time",
                            "unit": "second"
                        }
                    }
                },
            }
        },
        {"$sort": {"count": -1}},
    ]

    res = db.Activity.aggregate(p)
    top_2 = []
    while len(top_2) < 2:
        el = next(res)
        if el != None:
            if el["_id"]["y"] == 2008 and el["_id"]["m"] == 11:
                el["timeSpent"] = el["timeSpent"] / 3600
                top_2.append(el)

    print("\nTask 9b")
    tabulate_rows([[row["_id"]["uid"], row["count"], row["timeSpent"]] for row in top_2], [
                  "uid", "n_activities", "hours"])


def task_10(db: Database):
    """
    Find the total distance (in km) walked in 2008, by user with id = 112.
    """
    start_2008 = datetime(2008, 1, 1)
    end_2008 = datetime(2008, 12, 31, 23, 59, 59)
    activities_112_walk_2008 = db.Activity.find({"user_id": "112", "transportation_mode": "walk", "start_date_time": {"$gte": start_2008, "$lte": end_2008}})
    trackpoints = db.TrackPoint.find({"activity_id": {'$in': [activity["_id"] for activity in activities_112_walk_2008]}})

    dist = 0
    prev_tp = trackpoints[0]
    for tp in trackpoints[1:]:
        if tp["activity_id"] != prev_tp["activity_id"]:
            continue
        prev_coords = (float(prev_tp["latitude"]), float(prev_tp["longitude"]))
        new_coords = (float(tp["latitude"]), float(tp["longitude"]))
        dist += haversine(prev_coords, new_coords)
        prev_tp = tp
    print("\nTask 10")
    print(f"User_id: 112\nDistance in 2008: {dist}")


def task_11(db: Database):
    """
    Find the top 20 users who have gained the most altitude meters.
        ○ Output should be a table with (id, total meters gained per user).
        ○ Remember that some altitude - values are invalid
        ○ Tip: ∑(tp_{n}.altitude − tp_{n−1}.altitude), tp_{n}.altitude > tp_{n−1}.altitude
    """
    # Short helper function to sum altitude gains
    def gain_alt(alt1, alt2):
        # return (alt2 - alt1) if alt1 < alt2 else 0
        return (alt2 - alt1) * 0.3048 if alt1 < alt2 else 0 # Convert from feet to metres

    num_users = db.User.count_documents({})

    trackpoints = db.TrackPoint.aggregate([
        {"$match": {"altitude": {"$ne": -777}}},
        {"$sort": {"activity_id": 1, "_id": 1}}
    ], allowDiskUse=True)

    activities = defaultdict(dict)
    activities_from_db = db.Activity.find({})
    for a in activities_from_db:
        activities[a["_id"]] = a

    trackpoints = list(trackpoints)

    alt_per_user = [0] * num_users
    prev_tp = trackpoints[0]
    for tp in trackpoints[1:]:
        # Ensure same activity and user
        uid = int(activities[tp["activity_id"]]["user_id"])
        if tp["activity_id"] != prev_tp["activity_id"] or uid != int(activities[prev_tp["activity_id"]]["user_id"]):
            prev_tp = tp
            continue

        alt_gain = gain_alt(float(prev_tp["altitude"]), float(tp["altitude"]))
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
    n_users = db.User.count_documents({})
    trackpoints_activity_fields = db.TrackPoint.aggregate([
        {"$lookup": {
            "from": "Activity",
            "localField": "activity_id",
            "foreignField": "_id",
            "pipeline": [
                {"$project": {"id": 1, "user_id": 1}}
            ],
            "as": "activity_fields"}
         },
        {"$sort": {"activity_id": 1, "date_time": 1}
         },
    ], allowDiskUse=True)
    trackpoints = list(trackpoints_activity_fields)
    invalid_per_user = [0] * n_users

    invalid = False
    prev_tp = trackpoints[0]
    for tp in trackpoints[1:]:
        # Ensure same acitivity and user
        if invalid:
            if tp["activity_fields"][0]["_id"] != prev_tp["activity_fields"][0]["_id"]:
                invalid = False
                prev_tp = tp
            continue

        if tp["activity_fields"][0]["_id"] != prev_tp["activity_fields"][0]["_id"] or tp["activity_fields"][0]["user_id"] != prev_tp["activity_fields"][0]["user_id"]:
            prev_tp = tp
            continue

        time_diff = (tp["date_time"] - prev_tp["date_time"]).total_seconds()
        if time_diff >= 300:
            invalid = True
            invalid_per_user[int(tp["activity_fields"][0]["user_id"])] += 1

        prev_tp = tp

    print("\nTask 12")
    tabulate_rows([(i, invalid_per_user[i]) for i in range(n_users)], ["uid", "Invalid activities"])


def init_db(instance: Database_instance):
    collections = ["User", "Activity", "TrackPoint"]

    for collection in collections:
        instance.drop_collection(collection_name=collection)
        instance.create_collection(collection_name=collection)

    instance.show_collections()
    instance._fill_with_data()


def task_0(db: Database):
    """
    This prints out the first 10 documents in each collection for part 1
    """
    rows_user = [f"\u007b'_id': {x['_id']}, 'has_labels': {x['has_labels']}, 'activities': [...]\u007d" for x in db.User.find({}).limit(10)]
    rows_activity = [x for x in db.Activity.find({}).limit(10)]
    rows_trackpoint = [x for x in db.TrackPoint.find({}).limit(10)]
    print("\nUser:")
    for row in rows_user:
        print(row)
    print("\nActivity:")
    for row in rows_activity:
        print(row)
    print("\nTrackPoint:")
    for row in rows_trackpoint:
        print(row)


if __name__ == '__main__':
    db_instance = None
    try:
        while len(sys.argv) == 1:
            ans = input("Which task you want to run? [i: init db, a: all, 1..12: task 1..12]: ")
            if ans == "a" or ans == "i" or ans.isdigit() and int(ans) in range(1, 13):
                sys.argv.append(ans)

        db_instance = Database_instance()
        tasks = [task_1,
                 task_2,
                 task_3,
                 task_4,
                 task_5,
                 task_6,
                 task_7,
                 task_8,
                 task_9,
                 task_10,
                 task_11,
                 task_12,
                 task_0]


        # db_instance.show_collections()
        # Run tasks
        if sys.argv[1] == "i":
            init_db(db_instance)
        elif sys.argv[1] == "a":
            for task in tasks:
                task(db_instance.db)
        elif int(sys.argv[1]) in range(0, 13):
            tasks[int(sys.argv[1]) - 1](db_instance.db)
        else:
            print(f"Wrong task argument, argv = {sys.argv}")

    finally:
        if db_instance:
            db_instance.connection.close_connection()
