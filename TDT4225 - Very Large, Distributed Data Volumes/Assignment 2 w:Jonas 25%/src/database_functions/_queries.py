from dataclasses import dataclass
from enum import Enum
from database_functions._create_table_queries import CREATE_TABLE_QUERIES


@dataclass
class User:
    id: str
    has_labels: bool


class Table(Enum):
    User = "User"
    Activity = "Activity"
    TrackPoint = "TrackPoint"


class Queries():
    def create_table(self, table: Table):
        return CREATE_TABLE_QUERIES[table.value]

    def drop_table(self, table: Table):
        return f"DROP TABLE IF EXISTS `{table.value}`;"

    def SELECT(self, table: Table, id: str = None):
        return f"SELECT * FROM {table.value}{(' WHERE id=' + id + ';') if id else ';'}"

    def INSERT(self, table: Table, data: dict()):
        query = f"INSERT INTO {table.value} ({','.join([str(s) for s in data.keys()])}) VALUES{tuple([x for x in data.values()])};"
        return query.replace("None", 'NULL'.strip("\'"))

    def _chunks(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def INSERT_BATCH(self, table: Table, value_keys: list, values: list):
        assert len(values) % len(value_keys) == 0, "Wrong amount of values"
        res = f"INSERT INTO {table.value} ({','.join(value_keys)}) VALUES"
        for l in list(self._chunks(values, len(value_keys))):
            res += str(tuple(['NULL'.strip('\'') if x == None else x for x in l]))
            res += ","
        res = res[:-1]
        res += ";"
        return res

    def DELETE(self, table: Table, id: str):
        return f"DELETE FROM {table.value} WHERE id=='{id}';"


if __name__ == "__main__":
    print("CREATE_USER_TABLE:\n", Queries().create_table(Table.User))
    print("DROP_TABLE:\n", Queries().drop_table(Table.User))
    print("INSERT:\n", Queries().INSERT(Table.User, User(None, True).__dict__))
    print("GET_ALL:\n", Queries().SELECT(Table.User))
    print("SELECT:\n", Queries().SELECT(Table.User, id="000"))
    print("DELETE:\n", Queries().DELETE(Table.User, id="000"))
    print("INSERT BATCH:\n", Queries().INSERT_BATCH(Table.User, ["id", "has_labels", "Noe"], [1, True, "Noe", 2, True, 3, False, 4, True]))
