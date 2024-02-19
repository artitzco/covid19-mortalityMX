import pandas as pd


def nextWeekday(date, n):
    return date + pd.offsets.BDay(n)


def weekdayCounter(date1, date2):

    sgn = 1
    if date1 > date2:
        date1, date2 = date2, date1
        sgn = -1
    count = pd.date_range(start=date1, end=date2, freq='B').size
    return sgn * (count - 1) if count > 0 else count


def str_to_date(item, format='%d/%m/%Y'):
    return pd.to_datetime(item, format=format)


def date_to_str(date, format='%d/%m/%Y'):
    return date.strftime(format)


def format_time(seconds):
    days, sec = divmod(round(seconds), 86400)
    hors, sec = divmod(sec, 3600)
    min, sec = divmod(sec, 60)
    return ' '.join(f"{value}{unit}" for value, unit in zip(
        [days, hors, min, sec, round(seconds) % 1000], ['d', 'h', 'm', 's', 'ms']) if value > 0)
