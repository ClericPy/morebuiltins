import pickle
import sqlite3
import typing
from contextlib import closing, nullcontext

__all__ = ["KV"]

# keys may be of other types depending on the configured key_type;
# noqa: UP007 because requires-python >= 3.8 forbids the `X | Y` runtime syntax
_KeyType = typing.Union[str, int, float, bytes]  # noqa: UP007


def _escape_like(text) -> str:
    """Escape LIKE wildcards (% and _) so that a prefix matches literally.

    Non-str values (e.g. an int key of a KV configured with key_type
    "INTEGER") are converted with `str()` first, which is also how sqlite3
    would bind them to the LIKE pattern.
    """
    text = str(text)
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _prefix_condition(prefix, literal: bool) -> typing.Tuple[str, str]:
    """Build the LIKE condition and its pattern for a key-prefix filter.

    literal=False keeps the historical behaviour: `%` and `_` in the prefix are
    LIKE wildcards. literal=True escapes them, so the prefix matches literally.
    """
    if literal:
        return " WHERE k LIKE ? ESCAPE '\\'", _escape_like(prefix) + "%"
    return " WHERE k LIKE ?", f"{prefix}%"


class KV:
    NOT_SET = object()

    def __init__(
        self,
        database=":memory:",
        table="table1",
        encoder: typing.Tuple[
            str,
            typing.Optional[typing.Callable],
            typing.Optional[typing.Callable],
        ] = (
            "BLOB",
            pickle.dumps,
            pickle.loads,
        ),
        key_type="TEXT",
        **kwargs,
    ):
        """A key-value store using sqlite3.

        There are only 2 columns: k (TEXT PRIMARY KEY) and v (BLOB), and no extra
        indexes beyond the implicit primary-key index.

        In some cases, you may customize the encoder and decoder for the value,
        such as ("TEXT", json.dumps, json.loads) / ("INTEGER", None, None) / ("TEXT", None, None),
        even ("BLOB", zlib.compress, zlib.decompress), where the callables must be
        passed as objects, not strings.

        [WARN]: You can use KV as a dictionary-like object, but the thread safety needs to be handled by the user. Or use a better choice like `sqlitedict`.

        Args:
            database (str, optional): database file path. Defaults to ":memory:".
            table (str, optional): table name. Defaults to "table1".
            encoder (typing.Tuple[str, typing.Optional[typing.Callable], typing.Optional[typing.Callable]], optional): encoder for the value. Defaults to ("BLOB", pickle.dumps, pickle.loads).
            key_type (str, optional): key type. Defaults to "TEXT".

        Examples:

            >>> kv = KV()
            >>> kv.set("k1", "v1")
            1
            >>> len(kv)
            1
            >>> kv.get("k1")
            'v1'
            >>> kv.delete("k1")
            1
            >>> len(kv)
            0
            >>> kv.get("k1", "default")
            'default'
            >>> kv["k1"] = "v1"
            >>> kv["k1"]
            'v1'
            >>> del kv["k1"]
            >>> "k1" in kv
            False
            >>> kv["k1"] = "v1"
            >>> kv.pop("k1")
            'v1'
            >>> kv.pop("k1", "default")
            'default'
            >>> kv["k1"] = "v1"
            >>> kv.popitem()
            ('k1', 'v1')
            >>> kv.setdefault("k1", "v1")
            'v1'
            >>> kv.setdefault("k1", "default")
            'v1'
            >>> kv.update({"k1": "v1", "k2": "v2"})
            2
            >>> sorted(kv.items())
            [('k1', 'v1'), ('k2', 'v2')]
            >>> sorted(kv.keys())
            ['k1', 'k2']
            >>> list(kv.values(limit=1))
            ['v1']
            >>> list(kv.items(order="ASC"))
            [('k1', 'v1'), ('k2', 'v2')]
            >>> list(kv.items(order="DESC"))
            [('k2', 'v2'), ('k1', 'v1')]
            >>> kv.clear()
            2
            >>> kv.count()
            0
            >>> kv.set("k1", "v1")
            1
            >>> kv.count("k")
            1
            >>> # test int key
            >>> kv = KV(key_type="INTEGER")
            >>> kv.set(1, "v1")
            1
            >>> kv.get(1)
            'v1'
            >>> kv.delete(1)
            1
            >>> _ = [kv.set(i, i) for i in range(10000)]
            >>> kv.count()
            10000
            >>> kv[5000]
            5000
            >>> # test json value
            >>> import json
            >>> kv = KV(encoder=("TEXT", json.dumps, json.loads))
            >>> kv.set("k1", {"k": "v"})
            1
            >>> kv.get("k1")
            {'k': 'v'}
            >>> # test zlib compress value
            >>> import zlib
            >>> kv = KV(encoder=("BLOB", zlib.compress, zlib.decompress))
            >>> kv.set("k1", b"v1")
            1
            >>> kv.get("k1")
            b'v1'
        """
        self.database, self.table = (database, table)
        self.value_type, encoder_v, decoder_v = encoder
        self.encoder_v = encoder_v or self.nothing
        self.decoder_v = decoder_v or self.nothing
        self.key_type = key_type
        self.non_commit_ctx = nullcontext()
        self.init_conn(**kwargs)
        self.column_handlers = {
            ("k", "v"): lambda row: (row[0], self.decoder_v(row[1])),
            ("v",): lambda row: self.decoder_v(row[0]),
            ("k",): lambda row: row[0],
        }

    @staticmethod
    def _quote(identifier: str) -> str:
        """Quote an SQL identifier with backticks, doubling embedded backticks."""
        return "`" + str(identifier).replace("`", "``") + "`"

    @staticmethod
    def nothing(value):
        return value

    def init_conn(self, **kwargs):
        self.conn = sqlite3.connect(self.database, **kwargs)
        with self.conn:
            self.conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._quote(self.table)} "
                f"(k {self.key_type} PRIMARY KEY, v {self.value_type})",
            )

    def transaction_ctx(self, commit=True):
        return self.conn if commit else self.non_commit_ctx

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.close()

    def __contains__(self, k):
        return self.contains(k)

    def __getitem__(self, k):
        return self.get(k)

    def __len__(self):
        return self.count()

    def __delitem__(self, k):
        if not self.delete(k):
            raise KeyError(k)

    def __setitem__(self, k, v):
        self.set(k, v)

    def __del__(self):
        self.close()

    def close(self):
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()

    # read-only operations

    def recreate_table(self):
        """Recreate the table, keeping the configured key_type/value_type.

        The table is dropped and created again, so all existing rows are lost.
        """
        with self.conn:
            self.conn.execute(f"DROP TABLE IF EXISTS {self._quote(self.table)}")
            self.conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._quote(self.table)} "
                f"(k {self.key_type} PRIMARY KEY, v {self.value_type})"
            )

    def count(self, prefix="", literal=False) -> typing.Union[int, None]:
        """Return the number of keys in the table.

        Args:
            prefix (str, optional): key prefix. Defaults to "".
            literal (bool, optional): True matches the prefix literally (% and _
                are escaped); False keeps the historical LIKE semantics, where
                they act as wildcards. Defaults to False.

        Returns:
            typing.Union[int, None]: number of keys in the table.

        >>> kv = KV()
        >>> kv["a_b"] = 1
        >>> kv["axb"] = 2
        >>> kv.count(prefix="a_b"), kv.count(prefix="a_b", literal=True)
        (2, 1)
        """
        sql = f"SELECT COUNT(*) FROM {self._quote(self.table)}"  # noqa: S608  # nosec B608  # table is backtick-quoted by _quote (embedded backticks doubled); no values are interpolated
        args: tuple = ()
        if prefix:
            condition, pattern = _prefix_condition(prefix, literal)
            sql += condition
            args += (pattern,)
        with closing(self.conn.execute(sql, args)) as cursor:
            row = cursor.fetchone()
            if row is None:
                return None
            else:
                return row[0]

    def get(self, k: _KeyType, default=NOT_SET):
        """Return the value of the key k. If the key does not exist, return default or raise KeyError.

        Args:
            k (_KeyType): key, following the configured key_type.
            default (optional): default value. Defaults to NOT_SET and raise KeyError if the key does not exist.

        Raises:
            KeyError: key does not exist.

        Returns:
            _type_: value of the key k.
        """
        with closing(
            self.conn.execute(
                f"SELECT v FROM {self._quote(self.table)} WHERE k = ?",  # noqa: S608  # nosec B608  # quoted table identifier; the key is a ? placeholder
                (k,),
            )
        ) as cursor:
            row = cursor.fetchone()
            if row is None:
                if default is self.NOT_SET:
                    raise KeyError(k)
                return default
            else:
                return self.decoder_v(row[0])

    def items(
        self,
        prefix="",
        order: typing.Literal["ASC", "DESC", "asc", "desc", ""] = "",
        limit=-1,
        columns=("k", "v"),
        literal=False,
    ) -> typing.Generator:
        """Return a generator of items from the table.

        Args:
            prefix (str, optional): key prefix. Defaults to "".
            order (typing.Literal["ASC", "DESC", "asc", "desc", ""], optional): order of keys. Defaults to "" (no order).
            limit (int, optional): limit the number of key-value pairs. Defaults to -1.
            columns (tuple, optional): columns to select. Defaults to ("k", "v").
                Only three combinations are supported: ("k", "v") yields (key, value) tuples,
                ("k",) yields keys, ("v",) yields values.
            literal (bool, optional): True matches the prefix literally (% and _
                are escaped); False keeps the historical LIKE semantics, where
                they act as wildcards. Defaults to False.
        """
        _columns = ", ".join([self._quote(column) for column in columns])
        _column_handler = self.column_handlers[columns]
        sql = f"SELECT {_columns} FROM {self._quote(self.table)}"  # noqa: S608  # nosec B608  # both table and columns are backtick-quoted by _quote (embedded backticks doubled)
        args: tuple = ()
        if prefix:
            condition, pattern = _prefix_condition(prefix, literal)
            sql += condition
            args += (pattern,)
        if order:
            if order.upper() not in {"ASC", "DESC"}:
                raise ValueError(f"Invalid order: {order}")
            else:
                sql += f" ORDER BY k {order}"
        if limit >= 0:
            args += (limit,)
            sql += " LIMIT ?"
        with closing(self.conn.execute(sql, args)) as cursor:
            for row in cursor:
                yield _column_handler(row)

    def keys(
        self,
        prefix="",
        order: typing.Literal["ASC", "DESC", "asc", "desc", ""] = "",
        limit=-1,
        literal=False,
    ) -> typing.Generator:
        """Return a generator of keys.

        Args:
            prefix (str, optional): key prefix. Defaults to "".
            order (typing.Literal["ASC", "DESC", "asc", "desc", ""], optional): order of keys. Defaults to "" (no order).
            limit (int, optional): _description_. Defaults to -1.
            literal (bool, optional): True matches the prefix literally (% and _
                are escaped); False keeps the historical LIKE semantics, where
                they act as wildcards. Defaults to False.

        Raises:
            ValueError: invalid order.
        """
        yield from self.items(prefix=prefix, order=order, limit=limit, columns=("k",), literal=literal)

    def values(
        self,
        prefix="",
        order: typing.Literal["ASC", "DESC", "asc", "desc", ""] = "",
        limit=-1,
        literal=False,
    ) -> typing.Generator:
        """Return a generator of values.

        Args:
            prefix (str, optional): key prefix. Defaults to "".
            order (typing.Literal["ASC", "DESC", "asc", "desc", ""], optional): order of keys. Defaults to "" (no order).
            limit (int, optional): _description_. Defaults to -1.
            literal (bool, optional): True matches the prefix literally (% and _
                are escaped); False keeps the historical LIKE semantics, where
                they act as wildcards. Defaults to False.
        """
        yield from self.items(prefix=prefix, order=order, limit=limit, columns=("v",), literal=literal)

    def contains(self, k: _KeyType):
        """Return True if the key k exists in the table, otherwise False.

        Args:
            k (_KeyType): key, following the configured key_type.

        Returns:
            bool: True if the key k exists in the table, otherwise False.
        """
        with closing(
            self.conn.execute(
                f"SELECT 1 FROM {self._quote(self.table)} WHERE k = ?",  # noqa: S608  # nosec B608  # quoted table identifier; the key is a ? placeholder
                (k,),
            )
        ) as cursor:
            return cursor.fetchone() is not None

    # read & write operations, return the number of rows affected

    def set(self, k: _KeyType, v, commit=True) -> int:
        """Set the value of the key k.

        Args:
            k (_KeyType): key, following the configured key_type.
            v : value.
            commit (bool, optional): commit the transaction. Defaults to True.

        Returns:
            int: number of rows affected.
        """
        with self.transaction_ctx(commit):
            with closing(
                self.conn.execute(
                    f"REPLACE INTO {self._quote(self.table)} (k, v) VALUES (?, ?)",  # noqa: S608  # nosec B608  # quoted table identifier; key and value are ? placeholders
                    (k, self.encoder_v(v)),
                )
            ) as cursor:
                return cursor.rowcount

    def delete(self, k: _KeyType, commit=True) -> int:
        """Delete the key k.

        Args:
            k (_KeyType): key, following the configured key_type.
            commit (bool, optional): commit the transaction. Defaults to True.

        Returns:
            int: number of rows affected.
        """
        with self.transaction_ctx(commit):
            with closing(
                self.conn.execute(
                    f"DELETE FROM {self._quote(self.table)} WHERE k = ?",  # noqa: S608  # nosec B608  # quoted table identifier; the key is a ? placeholder
                    (k,),
                )
            ) as cursor:
                return cursor.rowcount

    def pop(self, k: _KeyType, default=NOT_SET, commit=True):
        """Delete the key k and return its value. If the key does not exist, return default or raise KeyError.

        Args:
            k (_KeyType): key, following the configured key_type.
            default (optional): default value. Defaults to NOT_SET and raise KeyError if the key does not exist.
            commit (bool, optional): commit the transaction. Defaults to True.

        Returns:
            value of the key k.
        """
        try:
            with self.transaction_ctx(commit):
                value = self.get(k, default=self.NOT_SET)
                self.delete(k, commit=False)
                return value
        except KeyError:
            if default is self.NOT_SET:
                raise
            return default

    def popitem(self, prefix="", commit=True, literal=False) -> typing.Union[None, typing.Tuple]:
        """Remove and return an arbitrary key-value pair. If the table is empty, return None.

        Args:
            prefix (str, optional): key prefix. Defaults to "".
            commit (bool, optional): commit the transaction. Defaults to True.
            literal (bool, optional): True matches the prefix literally (% and _
                are escaped); False keeps the historical LIKE semantics, where
                they act as wildcards. Defaults to False.

        Returns:
            typing.Union[None, typing.Tuple]: key-value pair.
        """
        with self.transaction_ctx(commit):
            for k in self.keys(prefix=prefix, limit=1, literal=literal):
                try:
                    value = self.pop(k, commit=False)
                    return k, value
                except KeyError:  # noqa: PERF203  # the key can be deleted between keys() and pop(), so each candidate needs its own handler
                    continue
        return None

    def setdefault(self, k, default, commit=True):
        """Return the value of the key k. If the key does not exist, set the key with the default value and return the default value.

        Args:
            k (str): key.
            default: default value.
            commit (bool, optional): commit the transaction. Defaults to True.

        Returns:
            value of the key k.
        """
        try:
            return self.get(k)
        except KeyError:
            with self.transaction_ctx(commit):
                self.set(k, default, commit=False)
            return default

    def update(self, data: typing.Dict[str, typing.Any], commit=True) -> int:
        """Update the table with the key-value pairs in data with executemany.

        A value of KV.NOT_SET deletes the corresponding key instead of writing.

        Args:
            data (typing.Dict[str, typing.Any]): key-value pairs.
            commit (bool, optional): commit the transaction. Defaults to True.

        Returns:
            int: number of rows affected (rows replaced plus rows deleted).
        """
        with self.transaction_ctx(commit):
            update_list = []
            delete_list = []
            for k, v in data.items():
                if v is self.NOT_SET:
                    delete_list.append((k,))
                else:
                    update_list.append((k, self.encoder_v(v)))
            rowcount = 0
            with closing(self.conn.cursor()) as cursor:
                if update_list:
                    rowcount += cursor.executemany(
                        f"REPLACE INTO {self._quote(self.table)} (k, v) VALUES (?, ?)",  # noqa: S608  # nosec B608  # quoted table identifier; key and value are ? placeholders
                        update_list,
                    ).rowcount
                if delete_list:
                    rowcount += cursor.executemany(
                        f"DELETE FROM {self._quote(self.table)} WHERE k = ?",  # noqa: S608  # nosec B608  # quoted table identifier; the key is a ? placeholder
                        delete_list,
                    ).rowcount
                return rowcount

    def copy(self, target: "KV", chunk_size=-1):
        """Copy the whole source database to the target KV with the sqlite3 backup API.

        This copies every table of the source database, not just the KV table;
        any existing tables in the target database, including the target's own
        working table, are replaced by the source's content. If the target's
        table name differs from the source's, the target KV is unusable
        afterwards: its own table no longer exists.

        Args:
            target (KV): target KV, its database is overwritten in place.
            chunk_size (int, optional): number of pages copied per step. Defaults to -1 (copy all pages at once).
        """
        with target.conn:
            return self.conn.backup(target.conn, pages=chunk_size)

    def clear(self, commit=True) -> int:
        """Delete all keys in the table.

        Args:
            commit (bool, optional): commit the transaction. Defaults to True.

        Returns:
            int: number of rows affected
        """
        with self.transaction_ctx(commit):
            with closing(
                self.conn.execute(f"DELETE FROM {self._quote(self.table)}")  # noqa: S608  # nosec B608  # table is backtick-quoted by _quote (embedded backticks doubled)
            ) as cursor:
                return cursor.rowcount

    def vacuum(self, commit=True) -> typing.Tuple[int, int]:
        """Run VACUUM to rebuild the database file.

        Args:
            commit (bool, optional): commit the transaction. Defaults to True.

        Returns:
            typing.Tuple[int, int]: old freelist count and new freelist count.
        """
        with self.transaction_ctx(commit):
            with closing(self.conn.cursor()) as cursor:
                old_count = cursor.execute("PRAGMA freelist_count").fetchone()[0]
                cursor.execute("VACUUM")
                new_count = cursor.execute("PRAGMA freelist_count").fetchone()[0]
                return old_count, new_count


if __name__ == "__main__":
    import doctest

    doctest.testmod()
