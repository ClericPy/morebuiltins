import re
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple, Union, Set, Generator


__all__ = ["ScheduleTimer", "Crontab"]


class ScheduleTimer:
    """The ScheduleTimer class facilitates the creation and evaluation of datetime patterns for scheduling purposes.

    It includes mechanisms to parse patterns involving logical operations (AND, OR) and comparison checks (equality, inequality, arithmetic, and custom range checks).

    Comparison Operators:

        Equality (= or ==): Tests for equality between datetime parts.
            Example: "hour=12" checks if it's exactly 12 o'clock.
        Inequality (!=): Ensures inequality between datetime parts.
            Example: "minute!=30" for minutes not being 30.
        Less Than (<): Requires the left datetime part to be less than the right.
            Example: "day<15" for days before the 15th.
        Less Than or Equal (<=): Allows the left datetime part to be less or equal to the right.
            Example: "month<=6" covers January to June.
        Greater Than (>): Ensures the left datetime part is greater than the right.
            Example: "hour>18" for evenings after 6 PM.
        Greater Than or Equal (>=): Allows the left datetime part to be greater or equal to the right.
            Example: "weekday>=MO" starts from Monday.
        Division Modulo (/): Divisibility check for the left digit by the right digit in datetime format.
            Example: "minute/15" checks for quarter hours.
        Range Inclusion (@): Confirms if the left time falls within any defined ranges in the right string.
            Example: "hour@9-11,13-15" for office hours.

    Logical Operators:
        AND (&): Both conditions must hold true.
            Example: "hour=12&minute=30" for exactly 12:30 PM.
        OR (; or |): At least one of the conditions must be true.
            Example: "hour=12;hour=18" for noon or 6 PM.
        Negation (!): Inverts the truth of the following condition. At the start of the pattern, the condition is negated.
            Example: "!hour=12" excludes noon.

    Demo:

    >>> start_date = datetime.strptime("2023-02-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    >>> list(ScheduleTimer.iter_datetimes("%M=05|%M=15", count=3, start_date=start_date, callback=str))
    ['2023-02-01 00:05:00', '2023-02-01 00:15:00', '2023-02-01 01:05:00']
    >>> list(ScheduleTimer.iter_datetimes("%H:%M=15:30", count=3, start_date=start_date, callback=str))
    ['2023-02-01 15:30:00', '2023-02-02 15:30:00', '2023-02-03 15:30:00']
    >>> list(ScheduleTimer.iter_datetimes("%H:%M=15:30&%d=15", count=3, start_date=start_date, callback=str))
    ['2023-02-15 15:30:00', '2023-03-15 15:30:00', '2023-04-15 15:30:00']
    >>> list(ScheduleTimer.iter_datetimes("%H:%M=15:30&%d>=15", count=3, start_date=start_date, callback=str))
    ['2023-02-15 15:30:00', '2023-02-16 15:30:00', '2023-02-17 15:30:00']
    >>> list(ScheduleTimer.iter_datetimes("%M@15-16&%d>=15", count=3, start_date=start_date, callback=str))
    ['2023-02-15 00:15:00', '2023-02-15 00:16:00', '2023-02-15 01:15:00']
    >>> list(ScheduleTimer.iter_datetimes("%M/15", count=5, start_date=start_date, callback=str))
    ['2023-02-01 00:00:00', '2023-02-01 00:15:00', '2023-02-01 00:30:00', '2023-02-01 00:45:00', '2023-02-01 01:00:00']
    """

    prefix_not = "!"
    split_or = re.compile(r";|\|")
    split_and = re.compile("&")
    split_calc = re.compile("(=+|!=|<=?|>=?|@|/)")
    magic_methods = {
        "=": "__eq__",
        "==": "__eq__",
        "!=": "__ne__",
        "<": "__lt__",
        "<=": "__le__",
        ">": "__gt__",
        ">=": "__ge__",
    }
    custom_methods = {"/": "divide", "@": "between"}
    step_choices = [
        ("%S", timedelta(seconds=1)),
        ("%M", timedelta(minutes=1)),
        ("%H", timedelta(hours=1)),
    ]
    min_delta = timedelta(minutes=1)

    def __init__(self):
        raise NotImplementedError(
            "Performance issue needs to be optimized, use the classmethod instead"
        )

    @staticmethod
    def divide(a: str, b: str):
        if a.isdigit() and b.isdigit():
            return int(a) % int(b) == 0
        else:
            return False

    @staticmethod
    def between(a: str, b: str):
        # many ranges
        range_regex = re.compile(r"^(\d+)-(\d+)$")
        for time_range in b.split(","):
            if a == time_range:
                return True
            else:
                match = range_regex.match(time_range)
                if match and int(a) >= int(match[1]) and int(a) <= int(match[2]):
                    return True
        return False

    @classmethod
    def get_step(cls, pattern: str):
        for step, delta in cls.step_choices:
            if step in pattern:
                if delta < cls.min_delta:
                    raise ValueError(
                        f"Step delta must be greater than {cls.min_delta.total_seconds()}, but given is {step}, or increase cls.min_delta but the performance will be worse"
                    )
                return timedelta(minutes=1)
        return timedelta(days=1)

    @classmethod
    def sort_pattern(cls, pattern: str):
        steps: Dict[str, timedelta] = {}
        for or_part in cls.split_or.split(pattern):
            and_parts = []
            for and_part in cls.split_and.split(or_part):
                steps[and_part] = cls.get_step(and_part)
                and_parts.append(and_part)
            and_parts.sort(key=lambda x: steps[x], reverse=True)
        return pattern, steps

    @classmethod
    def iter_datetimes(
        cls,
        pattern: str,
        count: int = 1,
        start_date: Optional[datetime] = None,
        max_tries: int = 1000000,
        callback: Callable = lambda x: x,
        yield_tries: bool = False,
    ) -> Generator:
        if start_date is None:
            _start_date = datetime.now().replace(microsecond=0)
        else:
            _start_date = start_date
        pattern_list, steps = cls.sort_pattern(pattern)
        min_step = min(steps.values())
        last_item = None
        matched = 0
        for tries in range(1, max_tries + 1):
            ok, miss_pattern = cls.is_at(pattern_list, _start_date)
            if ok:
                item = callback(_start_date)
                if not last_item or item != last_item:
                    matched += 1
                    if yield_tries:
                        yield tries, item
                    else:
                        yield item
                    if matched == count:
                        break
                step = min_step
            else:
                step = steps[miss_pattern]
            _start_date += step

    @classmethod
    def is_at(cls, pattern: Union[list, str], date: datetime) -> Tuple[bool, str]:
        if isinstance(pattern, str):
            pattern_list, _ = cls.sort_pattern(pattern)
        else:
            pattern_list = pattern
        miss_pattern: str = ""
        ok = False
        for or_pattern in cls.split_or.split(pattern_list):
            for and_pattern in cls.split_and.split(or_pattern):
                if and_pattern[0] == cls.prefix_not:
                    and_pattern = and_pattern[1:]
                    be_not = True
                else:
                    be_not = False
                _a, cmp, _b = cls.split_calc.split(and_pattern)
                if _a and _b:
                    a = date.strftime(_a)
                    b = date.strftime(_b)
                    magic_method = getattr(a, cls.magic_methods.get(cmp, ""), None)
                    if magic_method:
                        result = magic_method(b)
                    else:
                        result = getattr(cls, cls.custom_methods[cmp])(a, b)
                    if be_not:
                        if not result:
                            continue
                        else:
                            # no match break
                            miss_pattern = miss_pattern or and_pattern
                            break
                    elif result:
                        continue
                    else:
                        # no match break
                        miss_pattern = miss_pattern or and_pattern
                        break
                else:
                    # no match break
                    miss_pattern = miss_pattern or and_pattern
                    break
            else:
                # each and_pattern is hit
                ok = True
                break
        return ok, miss_pattern


class Crontab:
    """Crontab python parser.

    Demo:

    >>> start_date = datetime.strptime("2023-02-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    >>> list(Crontab.iter_datetimes("*/15 * * * *", count=10, start_date=start_date, callback=lambda i: i.strftime("%M")))
    ['00', '15', '30', '45', '00', '15', '30', '45', '00', '15']
    >>> list(Crontab.iter_datetimes("* * * * 2,4,6", count=10, start_date=start_date, callback=lambda i: i.strftime("%a")))
    ['Thu', 'Sat', 'Tue', 'Thu', 'Sat', 'Tue', 'Thu', 'Sat', 'Tue', 'Thu']
    >>> list(Crontab.iter_datetimes("0 0 11-19/4,8,30 * *", count=10, start_date=start_date, callback=lambda i: i.strftime("%m-%d")))
    ['02-08', '02-11', '02-15', '02-19', '03-08', '03-11', '03-15', '03-19', '03-30', '04-08']
    >>> list(Crontab.iter_datetimes("* * * * *", count=1, start_date=start_date))
    [datetime.datetime(2023, 2, 1, 0, 0)]
    >>> list(Crontab.iter_datetimes("5 4-5,6-9/2 5,6 * 3,5", count=3, start_date=start_date, callback=str))
    ['2023-04-05 04:05:00', '2023-04-05 05:05:00', '2023-04-05 06:05:00']
    >>> list(Crontab.iter_datetimes("0 0 1 8,9,10 *", count=3, start_date=start_date, callback=str, yield_tries=True))
    [(7, '2023-08-01 00:00:00'), (120, '2023-09-01 00:00:00'), (232, '2023-10-01 00:00:00')]
    >>> list(Crontab.iter_datetimes("0 0 1 11 *", count=3, start_date=start_date, callback=str, yield_tries=True))
    [(10, '2023-11-01 00:00:00'), (133, '2024-11-01 00:00:00'), (256, '2025-11-01 00:00:00')]
    """

    mins = list(range(0, 60))
    hours = list(range(0, 24))
    days = list(range(1, 32))
    months = list(range(1, 13))
    # crontab 0~7
    weeks = list(range(0, 8))
    # crontab: 0, 7 is Sunday
    # python: Monday is 0 and Sunday is 6
    wrap_cron_to_python = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}

    @classmethod
    def parse_field(cls, value: str, range_list: List[int]) -> Set[int]:
        """Parse a single cron field and return a list of matching values."""
        if value == "*":
            return set(range_list)
        results: set = set()
        valid_values = set(range_list)
        for val in value.split(","):
            if val == "*":
                results.update(range_list)
                break
            elif val.isdigit():
                results.add(int(val))
                continue
            if "/" in val:
                val, right = val.split("/")
                step = int(right)
            else:
                step = 1
            if val == "*":
                results.update(range(range_list[0], range_list[-1] + 1, step))
            elif "-" in val:
                start, end = map(int, val.split("-"))
                results.update(range(start, end + 1, step))
        return {i for i in results if i in valid_values}

    @classmethod
    def iter_datetimes(
        cls,
        pattern: str,
        count: int = 1,
        start_date: Optional[datetime] = None,
        max_tries: int = 1000000,
        callback: Callable = lambda x: x,
        yield_tries=False,
    ) -> Generator:
        if start_date is None:
            _start_date = datetime.now().replace(microsecond=0)
        else:
            _start_date = start_date
        _start_date = _start_date.replace(second=0)
        M, H, d, m, w = pattern.split(" ")
        mins = cls.parse_field(M, cls.mins)
        hours = cls.parse_field(H, cls.hours)
        days = cls.parse_field(d, cls.days)
        months = cls.parse_field(m, cls.months)
        weeks = cls.parse_field(w, cls.weeks)
        py_weekdays = {cls.wrap_cron_to_python[i] for i in weeks}
        tries = 0
        matched = 0
        last_match = None
        while tries < max_tries:
            tries += 1
            if _start_date.month not in months:
                # fast forward to next month with performance improvement
                temp = _start_date.replace(day=28, hour=0, minute=0) + timedelta(days=4)
                _start_date = temp.replace(day=1)
            elif (
                _start_date.weekday() not in py_weekdays or _start_date.day not in days
            ):
                _start_date = _start_date.replace(hour=0, minute=0) + timedelta(days=1)
            elif _start_date.hour not in hours:
                _start_date = _start_date.replace(minute=0) + timedelta(hours=1)
            elif _start_date.minute not in mins:
                _start_date += timedelta(minutes=1)
            else:
                item = callback(_start_date)
                if not last_match or item != last_match:
                    matched += 1
                    if yield_tries:
                        yield tries, item
                    else:
                        yield item
                    last_match = item
                    if matched == count:
                        break
                _start_date += timedelta(minutes=1)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
