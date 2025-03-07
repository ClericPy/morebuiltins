import pickle
import sqlite3
import typing
from contextlib import closing, nullcontext

__all__ = ["KV"]


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
        **kwargs,
    ):
        """A key-value store using sqlite3.

        There are only 2 columns: k (TEXT PRIMARY KEY) and v (BLOB), and without any index.

        In some cases, you may customize the encoder and decoder for the value,
        such as ("TEXT", "json.dumps", "json.loads") / ("INTEGER", None, None) / ("TEXT", None, None), even ("BLOB", zlib.compress, zlib.decompress).

        [WARN]: You can use KV as a dictionary-like object, but the thread safety needs to be handled by the user. Or use a better choice like `sqlitedict`.

        Args:
            database (str, optional): database file path. Defaults to ":memory:".
            table (str, optional): table name. Defaults to "table1".
            encoder (typing.Tuple[str, typing.Optional[typing.Callable], typing.Optional[typing.Callable]], optional): encoder for the value. Defaults to ("BLOB", pickle.dumps, pickle.loads).

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
        """
        self.database, self.table = (database, table)
        self.value_type, encoder_v, decoder_v = encoder
        self.encoder_v = encoder_v or self.nothing
        self.decoder_v = decoder_v or self.nothing
        self.non_commit_ctx = nullcontext()
        self.init_conn(**kwargs)
        self.column_handlers = {
            ("k", "v"): lambda row: (row[0], self.decoder_v(row[1])),
            ("v",): lambda row: self.decoder_v(row[0]),
            ("k",): lambda row: row[0],
        }

    @staticmethod
    def nothing(value):
        return value

    def init_conn(self, **kwargs):
        self.conn = sqlite3.connect(self.database, **kwargs)
        with self.conn:
            self.conn.execute(
                f"CREATE TABLE IF NOT EXISTS `{self.table}` (k TEXT PRIMARY KEY, v {self.value_type})",
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
        self.conn.close()

    # read-only operations

    def recreate_table(self):
        """Recreate the table."""
        with self.conn:
            self.conn.execute(f"DROP TABLE IF EXISTS `{self.table}`")
            self.conn.execute(
                f"CREATE TABLE IF NOT EXISTS `{self.table}` (k TEXT PRIMARY KEY, v BLOB)"
            )

    def count(self, prefix="") -> typing.Union[int, None]:
        """Return the number of keys in the table.

        Args:
            prefix (str, optional): key prefix. Defaults to "".

        Returns:
            typing.Union[int, None]: number of keys in the table.
        """
        sql = f"SELECT COUNT(*) FROM `{self.table}`"
        args: tuple = ()
        if prefix:
            sql += " WHERE k LIKE ?"
            args += (f"{prefix}%",)
        with closing(self.conn.execute(sql, args)) as cursor:
            row = cursor.fetchone()
            if row is None:
                return None
            else:
                return row[0]

    def get(self, k: str, default=NOT_SET):
        """Return the value of the key k. If the key does not exist, return default or raise KeyError.

        Args:
            k (str): key.
            default (optional): default value. Defaults to NOT_SET and raise KeyError if the key does not exist.

        Raises:
            KeyError: key does not exist.

        Returns:
            _type_: value of the key k.
        """
        with closing(
            self.conn.execute(f"SELECT v FROM `{self.table}` WHERE k = ?", (k,))
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
    ) -> typing.Generator:
        """Return a generator of key-value pairs.

        Args:
            prefix (str, optional): key prefix. Defaults to "".
            order (typing.Literal["ASC", "DESC", "asc", "desc", ""], optional): order of keys. Defaults to "" (no order).
            limit (int, optional): limit the number of key-value pairs. Defaults to -1.
        """
        _columns = ", ".join([f"`{column}`" for column in columns])
        _column_handler = self.column_handlers[columns]
        sql = f"SELECT {_columns} FROM `{self.table}`"
        args: tuple = ()
        if prefix:
            args += (f"{prefix}%",)
            sql += " WHERE k LIKE ?"
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
    ) -> typing.Generator:
        """Return a generator of keys.

        Args:
            prefix (str, optional): key prefix. Defaults to "".
            order (typing.Literal["ASC", "DESC", "asc", "desc", ""], optional): order of keys. Defaults to "" (no order).
            limit (int, optional): _description_. Defaults to -1.

        Raises:
            ValueError: invalid order.
        """
        yield from self.items(prefix=prefix, order=order, limit=limit, columns=("k",))

    def values(
        self,
        prefix="",
        order: typing.Literal["ASC", "DESC", "asc", "desc", ""] = "",
        limit=-1,
    ) -> typing.Generator:
        """Return a generator of values.

        Args:
            prefix (str, optional): key prefix. Defaults to "".
            order (typing.Literal["ASC", "DESC", "asc", "desc", ""], optional): order of keys. Defaults to "" (no order).
            limit (int, optional): _description_. Defaults to -1.
        """
        yield from self.items(prefix=prefix, order=order, limit=limit, columns=("v",))

    def contains(self, k: str):
        """Return True if the key k exists in the table, otherwise False.

        Args:
            k (str): key.

        Returns:
            bool: True if the key k exists in the table, otherwise False.
        """
        with closing(
            self.conn.execute(f"SELECT 1 FROM `{self.table}` WHERE k = ?", (k,))
        ) as cursor:
            return cursor.fetchone() is not None

    # read & write operations, return the number of rows affected

    def set(self, k: str, v, commit=True) -> int:
        """Set the value of the key k.

        Args:
            k (str): key.
            v : value.
            commit (bool, optional): commit the transaction. Defaults to True.

        Returns:
            int: number of rows affected.
        """
        with self.transaction_ctx(commit):
            with closing(
                self.conn.execute(
                    f"REPLACE INTO `{self.table}` (k, v) VALUES (?, ?)",
                    (k, self.encoder_v(v)),
                )
            ) as cursor:
                return cursor.rowcount

    def delete(self, k: str, commit=True) -> int:
        """Delete the key k.

        Args:
            k (str): key.
            commit (bool, optional): commit the transaction. Defaults to True.

        Returns:
            int: number of rows affected.
        """
        with self.transaction_ctx(commit):
            with closing(
                self.conn.execute(f"DELETE FROM `{self.table}` WHERE k = ?", (k,))
            ) as cursor:
                return cursor.rowcount

    def pop(self, k: str, default=NOT_SET, commit=True):
        """Delete the key k and return its value. If the key does not exist, return default or raise KeyError.

        Args:
            k (str): key.
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

    def popitem(self, prefix="", commit=True) -> typing.Union[None, typing.Tuple]:
        """Remove and return an arbitrary key-value pair. If the table is empty, return None.

        Args:
            prefix (str, optional): key prefix. Defaults to "".
            commit (bool, optional): commit the transaction. Defaults to True.

        Returns:
            typing.Union[None, typing.Tuple]: key-value pair.
        """
        with self.transaction_ctx(commit):
            for k in self.keys(prefix=prefix, limit=1):
                try:
                    value = self.pop(k, commit=False)
                    return k, value
                except KeyError:
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

        Args:
            data (typing.Dict[str, typing.Any]): key-value pairs.
            commit (bool, optional): commit the transaction. Defaults to True.

        Returns:
            int: number of rows affected.
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
                        f"REPLACE INTO `{self.table}` (k, v) VALUES (?, ?)", update_list
                    ).rowcount
                if delete_list:
                    rowcount += cursor.executemany(
                        f"DELETE FROM `{self.table}` WHERE k = ?", delete_list
                    ).rowcount
                return rowcount

    def copy(self, target: "KV", chunk_size=-1):
        """Copy the table to the target KV.

        Args:
            target (KV): target KV.
            chunk_size (int, optional): chunk size. Defaults to -1.
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
            with closing(self.conn.execute(f"DELETE FROM `{self.table}`")) as cursor:
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
