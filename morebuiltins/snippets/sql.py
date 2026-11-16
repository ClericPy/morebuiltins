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
        """Clear the free pages of the database to shrink the file size.

        `PRAGMA incremental_vacuum` is a silent no-op unless `auto_vacuum`
        is INCREMENTAL (value 2), so for other databases run `VACUUM` first
        to release the free pages.

        For a non-incremental database this calls `db.commit()` and then a
        full `VACUUM`, so note that:

        - it commits any pending transaction of the caller;
        - it is O(database size) and temporarily needs free disk space
          roughly equal to the database size to rebuild the file;
        - it fails ("cannot VACUUM - SQL statements in progress") while the
          connection still has an unfinished cursor/statement.

        Returns a tuple of (freelist_count_before, freelist_count_after):
        after < before means the free pages were cleared.
        """
        before_count = db.execute("PRAGMA freelist_count").fetchone()[0]
        auto_vacuum = db.execute("PRAGMA auto_vacuum").fetchone()[0]
        if auto_vacuum != 2:  # not INCREMENTAL
            # VACUUM cannot run inside a transaction
            db.commit()
            db.execute("VACUUM")
        sql = f"PRAGMA incremental_vacuum({chunk_size})"
        for _ in range(tries):
            db.execute(sql)
            db.commit()
        new_count = db.execute("PRAGMA freelist_count").fetchone()[0]
        return before_count, new_count

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

        SQLite only accepts AUTOINCREMENT on a single-column INTEGER primary
        key, so anything else is rejected instead of emitting invalid SQL:
        >>> SqliteSQL.create_table("user2", User2.__annotations__, primary_key=("id", "name"), autoincrement=True)
        Traceback (most recent call last):
        ...
        ValueError: autoincrement requires a single-column primary key, got ('id', 'name')
        >>> SqliteSQL.create_table("user2", User2.__annotations__, primary_key="name", autoincrement=True)
        Traceback (most recent call last):
        ...
        ValueError: autoincrement requires an INTEGER primary key, got <class 'str'> for 'name'
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
        pk_clause = cls._primary_key_clause(data_types, primary_key, autoincrement)
        if pk_clause:
            table_sql += f"{pk_clause},"

        table_sql = table_sql.rstrip(", ")
        table_sql += ");"
        result = [table_sql]
        result.extend(cls.create_index(table, index, unique=True) for index in unique_indexes or ())
        result.extend(cls.create_index(table, index, unique=False) for index in indexes or ())
        return result

    @staticmethod
    def _primary_key_clause(data_types, primary_key, autoincrement):
        """Build the ` PRIMARY KEY (...)` clause, validating AUTOINCREMENT.

        >>> SqliteSQL._primary_key_clause({"id": int}, "id", True)
        ' PRIMARY KEY (id AUTOINCREMENT)'
        >>> SqliteSQL._primary_key_clause({"id": int}, ("id",), False)
        ' PRIMARY KEY (id)'
        >>> SqliteSQL._primary_key_clause({"id": int}, (), False)
        ''
        """
        if not primary_key:
            return ""
        if isinstance(primary_key, str):
            primary_key = (primary_key,)
        pk = ", ".join(primary_key)
        if autoincrement:
            # `PRIMARY KEY (a, b AUTOINCREMENT)` is invalid SQL, and
            # AUTOINCREMENT only works on an INTEGER primary key
            if len(primary_key) != 1:
                raise ValueError(f"autoincrement requires a single-column primary key, got {primary_key!r}")
            pk_column = primary_key[0]
            pk_type = data_types.get(pk_column)
            if pk_type is not int:
                raise ValueError(f"autoincrement requires an INTEGER primary key, got {pk_type!r} for {pk_column!r}")
            pk += " AUTOINCREMENT"
        return f" PRIMARY KEY ({pk})"

    @classmethod
    def create_index(cls, table: str, index: typing.Union[str, typing.Sequence[str]], unique=False):
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
        return f"CREATE {uniq}INDEX IF NOT EXISTS `{idx_name}` ON {table} ({idx_value});"

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
        'INSERT INTO user (`id`, `name`, `age`, `score`, `image`) VALUES (:id, :name, :age, :score, :image) ON CONFLICT(`id`) DO UPDATE SET (`id`, `name`, `age`, `score`, `image`) = (:id, :name, :age, :score, :image);'
        >>> SqliteSQL.upsert("user", user.keys(), primary_key=("id", "name"))
        'INSERT INTO user (`id`, `name`, `age`, `score`, `image`) VALUES (:id, :name, :age, :score, :image) ON CONFLICT(`id`, `name`) DO UPDATE SET (`id`, `name`, `age`, `score`, `image`) = (:id, :name, :age, :score, :image);'

        Behavior changed: the old version generated `INSERT OR REPLACE INTO ... ON CONFLICT`,
        where `OR REPLACE` took precedence over the upsert clause for conflicts on other
        UNIQUE constraints/indexes, silently deleting the conflicting rows. The standard
        `INSERT INTO ... ON CONFLICT(...) DO UPDATE` now updates the row conflicting on the
        primary key, and raises `sqlite3.IntegrityError` for conflicts on other UNIQUE
        constraints instead of deleting rows.
        """
        _keys = ", ".join([f"`{k}`" for k in keys])
        _values = ", ".join([f":{k}" for k in keys])
        if isinstance(primary_key, str):
            primary_key = (primary_key,)
        pk = ", ".join([f"`{k}`" for k in primary_key])
        return (
            f"INSERT INTO {table} ({_keys}) VALUES ({_values}) ON CONFLICT({pk}) DO UPDATE SET ({_keys}) = ({_values});"  # noqa: S608  # nosec B608  # keys are backtick-quoted and values are :name placeholders; table is a trusted identifier for this generator
        )

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
        result = f"UPDATE {table} SET ({_keys}) = ({_values})"  # noqa: S608  # nosec B608  # keys are backtick-quoted and values are :name placeholders; table is a trusted identifier for this generator
        if where:
            result += f" WHERE {where}"
        result += ";"
        return result

    @classmethod
    def delete(cls, table: str, where: str = "", limit=0):
        r"""Generate delete sql.

        The row limit is applied through a `rowid` subquery because the
        SQLite build shipped with CPython does not support `DELETE ... LIMIT`.
        `where` appears only once, inside the subquery, so a top-level `OR`
        cannot escape the limit.

        Not supported: `WITHOUT ROWID` tables (no rowid to select), and
        tables that declare a plain `rowid` column (the subquery resolves
        `rowid` to that column instead of the implicit row id).

        >>> SqliteSQL.delete("user")
        'DELETE FROM user;'
        >>> SqliteSQL.delete("user", limit=100)
        'DELETE FROM user WHERE rowid IN (SELECT rowid FROM user ORDER BY rowid LIMIT 100);'
        >>> SqliteSQL.delete("user", where="`id` < 10", limit=5)
        'DELETE FROM user WHERE rowid IN (SELECT rowid FROM user WHERE (`id` < 10) ORDER BY rowid LIMIT 5);'

        Executed against a real table:

        >>> import sqlite3
        >>> db = sqlite3.connect(":memory:")
        >>> _ = db.execute("CREATE TABLE user(id INTEGER PRIMARY KEY, name TEXT)")
        >>> _ = db.executemany("INSERT INTO user(name) VALUES (?)", [("a",), ("b",), ("c",)])
        >>> # no where: the limit itself decides how many rows are deleted
        >>> db.execute(SqliteSQL.delete("user", limit=2)).rowcount
        2
        >>> db.execute("SELECT COUNT(*) FROM user").fetchone()[0]
        1
        >>> _ = db.execute("INSERT INTO user(name) VALUES ('b'), ('c'), ('d')")
        >>> # where with a top-level OR: the limit applies to the whole where clause
        >>> db.execute(SqliteSQL.delete("user", where="name = 'b' OR name = 'c'", limit=1)).rowcount
        1
        >>> db.execute("SELECT COUNT(*) FROM user").fetchone()[0]
        3
        >>> # without a limit the where clause is used as-is
        >>> db.execute(SqliteSQL.delete("user", where="name = 'd'")).rowcount
        1
        >>> db.execute("SELECT COUNT(*) FROM user").fetchone()[0]
        2
        """
        result = f"DELETE FROM {table}"  # noqa: S608  # nosec B608  # table is a trusted identifier for this generator; limit is an int and rows are matched by rowid, not by interpolated values
        if limit:
            sub_query = f"SELECT rowid FROM {table}"  # noqa: S608  # nosec B608  # same trusted table identifier; the where clause stays raw SQL by API design
            if where:
                sub_query += f" WHERE ({where})"
            sub_query += f" ORDER BY rowid LIMIT {limit}"
            result += f" WHERE rowid IN ({sub_query})"
        elif where:
            result += f" WHERE {where}"
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
