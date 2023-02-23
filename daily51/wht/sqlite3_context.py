"""
Connect to sqlite3 database like a file with python context object
"""
import sqlite3
from .file_system import utouch


class BaseDB:
    """Basic sqlite3 context object wrapper."""

    @classmethod
    def get_default_options(cls):
        return {
            "timeout": 300
        }

    def __init__(self, db_filepath, **kwargs):
        self.db_filepath = db_filepath
        self.connect_options = self.get_default_options()
        self.connect_options.update(kwargs)
        self.conn = None
        self.cursor = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_filepath, **self.connect_options)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, exit_type, exit_value, traceback):
        self.conn.commit()
        self.cursor.close()
        self.conn.close()

    def __call__(self, *args) -> sqlite3.Cursor:
        return self.cursor.execute(*args)

    def create(self):
        """execute ``CREATE TABLE`` sql"""
        raise NotImplementedError(
            "implement create method or execute CREATE sql yourself")

    @classmethod
    def init_db_filepath(cls, db_filepath):
        utouch(db_filepath)
        with cls(db_filepath) as db:
            db.create()
