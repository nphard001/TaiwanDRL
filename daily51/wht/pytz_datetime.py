"""
Datetime utils based on ``pytz`` library
"""
import datetime
import pytz

pytz_tw = pytz.timezone('Asia/Taipei')
pytz_us = pytz.timezone('US/Eastern')
pytz_utc = pytz.utc
python_utc = datetime.timezone.utc


def utc_now() -> datetime.datetime:
    """
    Time-zone-aware version of "now"
    Django's auto_now implemented it
    Source: django.timezone.now
    """
    return datetime.datetime.utcnow().replace(tzinfo=pytz_utc)


def tw_now() -> datetime.datetime:
    return pytz_tw.fromutc(datetime.datetime.utcnow())


def naive_datetime(year_or_strptime,
                   month=None, day=None, hour=0, minute=0, second=0,
                   microsecond=0, format=None) -> datetime.datetime:
    if isinstance(year_or_strptime, str):
        format = '%Y%m%d' if format is None else format
        return datetime.datetime.strptime(year_or_strptime, format)
    return datetime.datetime(year_or_strptime, month, day, hour, minute, second, microsecond)


def utc_datetime(*args, **kwargs) -> datetime.datetime:
    """sugar to construct utc datetime instance"""
    return naive_datetime(*args, **kwargs).replace(tzinfo=python_utc)


def tw_datetime(*args, **kwargs) -> datetime.datetime:
    """sugar to construct pytz style tw datetime instance"""
    return pytz_tw.localize(naive_datetime(*args, **kwargs))


def timestamp_to_utc(timestamp_int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(timestamp_int, python_utc)


def timestamp_to_tw(timestamp_int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(timestamp_int, pytz_tw)
