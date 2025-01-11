import re
import sqlite3
import typing

__all__ = ["SqliteSQL"]


class SqliteSQL:
    """Sqlite SQL generator"""

    @staticmethod
    def title_to_snake_case(string: str):
        """Title string to snake case
        >>> SqliteSQL.title_to_snake_case("HelloWorld")
        'hello_world'
        >>> SqliteSQL.title_to_snake_case("HelloHTMLWorldPeace")
        'hello_html_world_peace'
        """
        return re.sub(
            r"([A-Z]+)([A-Z][a-z])",
            r"\1_\2",
            re.sub(r"([a-z\d])([A-Z])", r"\1_\2", string),
        ).lower()

    @classmethod
    def clear_free_pages(cls, db: sqlite3.Connection, tries=1, chunk_size=100):
        db.execute("PRAGMA auto_vacuum = INCREMENTAL;")
        sql = "PRAGMA incremental_vacuum({chunk_size})".format(chunk_size=chunk_size)
        for _ in range(tries):
            db.execute(sql)
            db.commit()
        new_count = db.execute("PRAGMA freelist_count").fetchone()[0]
        return new_count

    @classmethod
    def create_table(
        cls,
        table: str,
        data_types: typing.Dict[str, typing.Type],
        primary_key: typing.Union[tuple, str] = (),
        unique_indexes: typing.Optional[list] = None,
        indexes: typing.Optional[list] = None,
        autoincrement=False,
        strict=True,
    ) -> list:
        r"""

        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class User:
        ...     id: int
        ...     name: str
        ...     age: int
        ...     score: float
        ...     image: bytes
        ...     primary_key = "id"
        ...     autoincrement = True
        ...     indexes = [["age", "score"], "score"]
        ...     unique_indexes = ["name", "image"]
        >>> SqliteSQL.create_table("user", User.__annotations__)
        ['CREATE TABLE IF NOT EXISTS user(`id` INTEGER,`name` TEXT,`age` INTEGER,`score` REAL,`image` BLOB);']
        >>> SqliteSQL.create_table("user", User.__annotations__, primary_key=User.primary_key, unique_indexes=User.unique_indexes, indexes=User.indexes, autoincrement=User.autoincrement)
        ['CREATE TABLE IF NOT EXISTS user(`id` INTEGER,`name` TEXT,`age` INTEGER,`score` REAL,`image` BLOB, PRIMARY KEY (id AUTOINCREMENT));', 'CREATE UNIQUE INDEX IF NOT EXISTS `idx_name` ON user (`name`);', 'CREATE UNIQUE INDEX IF NOT EXISTS `idx_image` ON user (`image`);', 'CREATE INDEX IF NOT EXISTS `idx_age_score` ON user (`age`, `score`);', 'CREATE INDEX IF NOT EXISTS `idx_score` ON user (`score`);']
        >>> from typing import TypedDict
        >>> class User2(TypedDict):
        ...     id: int
        ...     name: str
        >>> SqliteSQL.create_table("user2", User2.__annotations__)
        ['CREATE TABLE IF NOT EXISTS user2(`id` INTEGER,`name` TEXT);']
        >>> SqliteSQL.create_table("user2", User2.__annotations__, primary_key=("id", "name"), unique_indexes=["name"])
        ['CREATE TABLE IF NOT EXISTS user2(`id` INTEGER,`name` TEXT, PRIMARY KEY (id, name));', 'CREATE UNIQUE INDEX IF NOT EXISTS `idx_name` ON user2 (`name`);']
        """
        type_mappings = {str: "TEXT", int: "INTEGER", float: "REAL", bytes: "BLOB"}
        table_sql = f"CREATE TABLE IF NOT EXISTS {table}"
        table_sql += "("
        if strict:
            not_supported = set(data_types.values()) - set(type_mappings.keys())
            if not_supported:
                raise TypeError(f"Unsupported types: {not_supported}")
        for key, value in data_types.items():
            tp = type_mappings.get(value, "TEXT")
            table_sql += f"`{key}` {tp},"
        if primary_key:
            if isinstance(primary_key, str):
                primary_key = (primary_key,)
            pk = ", ".join(primary_key)
            if autoincrement:
                pk += " AUTOINCREMENT"
            table_sql += f" PRIMARY KEY ({pk}),"

        table_sql = table_sql.rstrip(", ")
        table_sql += ");"
        result = [table_sql]
        if unique_indexes:
            for index in unique_indexes:
                result.append(cls.create_index(table, index, unique=True))
        if indexes:
            for index in indexes:
                result.append(cls.create_index(table, index, unique=False))
        return result

    @classmethod
    def create_index(
        cls, table: str, index: typing.Union[str, typing.Sequence[str]], unique=False
    ):
        r"""
        >>> SqliteSQL.create_index("user", "name")
        'CREATE INDEX IF NOT EXISTS `idx_name` ON user (`name`);'
        >>> SqliteSQL.create_index("user", ["name", "age"])
        'CREATE INDEX IF NOT EXISTS `idx_name_age` ON user (`name`, `age`);'
        >>> SqliteSQL.create_index("user", ["name", "age"], unique=True)
        'CREATE UNIQUE INDEX IF NOT EXISTS `idx_name_age` ON user (`name`, `age`);'
        """
        if isinstance(index, str):
            index = (index,)
        idx_name = "idx_" + "_".join(index)
        idx_value = ", ".join([f"`{i}`" for i in index])
        uniq = "UNIQUE " if unique else ""
        return (
            f"CREATE {uniq}INDEX IF NOT EXISTS `{idx_name}` ON {table} ({idx_value});"
        )

    @classmethod
    def insert(
        cls,
        table: str,
        keys: typing.Sequence,
        method: typing.Literal["insert", "replace"],
    ):
        """Generate insert sql.

        >>> user = dict(id=1, name="name", age=1, score=1.0, image=b"image")
        >>> SqliteSQL.insert("user", user.keys(), method="insert")
        'INSERT INTO user (`id`, `name`, `age`, `score`, `image`) VALUES (:id, :name, :age, :score, :image);'
        >>> SqliteSQL.insert("user", user.keys(), method="replace")
        'REPLACE INTO user (`id`, `name`, `age`, `score`, `image`) VALUES (:id, :name, :age, :score, :image);'
        """
        _keys = ", ".join([f"`{k}`" for k in keys])
        _values = ", ".join([f":{k}" for k in keys])
        return f"{method.upper()} INTO {table} ({_keys}) VALUES ({_values});"

    @classmethod
    def upsert(
        cls,
        table: str,
        keys: typing.Sequence,
        primary_key: typing.Union[str, typing.Sequence[str]],
    ):
        """Generate upsert(insert or update) sql.

        >>> user = dict(id=1, name="name", age=1, score=1.0, image=b"image")
        >>> SqliteSQL.upsert("user", user.keys(), primary_key="id")
        'INSERT OR REPLACE INTO user (`id`, `name`, `age`, `score`, `image`) VALUES (:id, :name, :age, :score, :image) ON CONFLICT(`id`) DO UPDATE SET (`id`, `name`, `age`, `score`, `image`) = (:id, :name, :age, :score, :image);'
        >>> SqliteSQL.upsert("user", user.keys(), primary_key=("id", "name"))
        'INSERT OR REPLACE INTO user (`id`, `name`, `age`, `score`, `image`) VALUES (:id, :name, :age, :score, :image) ON CONFLICT(`id`, `name`) DO UPDATE SET (`id`, `name`, `age`, `score`, `image`) = (:id, :name, :age, :score, :image);'
        """
        _keys = ", ".join([f"`{k}`" for k in keys])
        _values = ", ".join([f":{k}" for k in keys])
        if isinstance(primary_key, str):
            primary_key = (primary_key,)
        pk = ", ".join([f"`{k}`" for k in primary_key])
        return f"INSERT OR REPLACE INTO {table} ({_keys}) VALUES ({_values}) ON CONFLICT({pk}) DO UPDATE SET ({_keys}) = ({_values});"

    @classmethod
    def update(cls, table: str, update_keys: typing.Sequence[str], where: str = ""):
        """Generate update sql.

        >>> user = dict(id=1, name="name", age=1, score=1.0, image=b"image")
        >>> SqliteSQL.update("user", user.keys(), where="")
        'UPDATE user SET (`id`, `name`, `age`, `score`, `image`) = (:id, :name, :age, :score, :image);'
        >>> SqliteSQL.update("user", ("name", "score"), where="`id` = 10")
        'UPDATE user SET (`name`, `score`) = (:name, :score) WHERE `id` = 10;'
        """
        _keys = ", ".join([f"`{k}`" for k in update_keys])
        _values = ", ".join([f":{k}" for k in update_keys])
        result = f"UPDATE {table} SET ({_keys}) = ({_values})"
        if where:
            result += f" WHERE {where}"
        result += ";"
        return result

    @classmethod
    def delete(cls, table: str, where: str = "", limit=0):
        r"""Generate delete sql.

        >>> SqliteSQL.delete("user")
        'DELETE FROM user;'
        >>> SqliteSQL.delete("user", limit=100)
        'DELETE FROM user LIMIT 100;'
        >>> SqliteSQL.delete("user", where="`id` < 10")
        'DELETE FROM user WHERE `id` < 10;'
        """
        result = f"DELETE FROM {table}"
        if where:
            result += f" WHERE {where}"
        if limit:
            result += f" LIMIT {limit}"
        result += ";"
        return result


# class Q:
#     """[WIP]
#     Where clause generator.
#     Q("name").IN("John", "Tom", "Jerry").AND("age").BETWEEN(20, 30).OR("city").IN("New York", "Los Angeles").str()
#     # name IN ('John', 'Tom', 'Jerry') AND age BETWEEN 20 AND 30 OR city IN ('New York', 'Los Angeles')
#     """


# class KVSqlite:
#     """[WIP]
#     Sqlite KV store. Production environment recommends using `sqlitedict`."""


if __name__ == "__main__":
    from doctest import testmod

    testmod()
