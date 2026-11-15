import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

__all__ = ["ScheduleTimer", "Crontab"]


class ScheduleTimer:
    """The ScheduleTimer class facilitates the creation and evaluation of datetime patterns for scheduling purposes.

    It includes mechanisms to parse patterns involving logical operations (AND, OR) and
    comparison checks (equality, inequality, arithmetic, and custom range checks).

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
        Negation (!): Inverts the truth of the following condition. At the start of
            the pattern, the condition is negated.
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
    >>> noon = datetime.strptime("2023-02-01 12:00:00", "%Y-%m-%d %H:%M:%S")
    >>> list(ScheduleTimer.iter_datetimes("!%H=12", count=2, start_date=noon, callback=str))
    ['2023-02-01 13:00:00', '2023-02-01 13:01:00']
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
                        f"Step delta must be greater than "
                        f"{cls.min_delta.total_seconds()}, but given is {step}, or "
                        f"increase cls.min_delta but the performance will be worse"
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
            # naive local time is the scheduling contract, so no tzinfo here
            _start_date = datetime.now().replace(microsecond=0)  # noqa: DTZ005
        else:
            _start_date = start_date
        pattern_list, steps = cls.sort_pattern(pattern)
        min_step = min(steps.values())
        last_item: Any = None
        matched = 0
        for tries in range(1, max_tries + 1):
            ok, miss_pattern = cls.is_at(pattern_list, _start_date)
            if ok:
                item = callback(_start_date)
                if last_item is None or item != last_item:
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
    def is_at(cls, pattern: str, date: datetime) -> Tuple[bool, str]:
        pattern_list, _ = cls.sort_pattern(pattern)
        miss_pattern: str = ""
        for or_pattern in cls.split_or.split(pattern_list):
            for and_pattern in cls.split_and.split(or_pattern):
                if cls._eval_and_pattern(and_pattern, date):
                    continue
                # no match break
                miss_pattern = miss_pattern or and_pattern
                break
            else:
                # each and_pattern is hit
                return True, miss_pattern
        return False, miss_pattern

    @classmethod
    def _eval_and_pattern(cls, and_pattern: str, date: datetime) -> bool:
        """Evaluate a single AND part, negated or not."""
        be_not = and_pattern.startswith(cls.prefix_not)
        if be_not:
            and_pattern = and_pattern[len(cls.prefix_not) :]
        _a, cmp, _b = cls.split_calc.split(and_pattern)
        if not (_a and _b):
            return False
        a = date.strftime(_a)
        b = date.strftime(_b)
        magic_method = getattr(a, cls.magic_methods.get(cmp, ""), None)
        if magic_method:
            result = magic_method(b)
        else:
            result = getattr(cls, cls.custom_methods[cmp])(a, b)
        return not result if be_not else result


class Crontab:
    """Crontab python parser.

    Demo:

    >>> start_date = datetime.strptime("2023-02-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    >>> list(Crontab.iter_datetimes("*/15 * * * *", count=10, start_date=start_date,
    ...     callback=lambda i: i.strftime("%M")))
    ['00', '15', '30', '45', '00', '15', '30', '45', '00', '15']
    >>> list(Crontab.iter_datetimes("* * * * 2,4,6", count=10, start_date=start_date,
    ...     callback=lambda i: i.strftime("%a")))
    ['Thu', 'Sat', 'Tue', 'Thu', 'Sat', 'Tue', 'Thu', 'Sat', 'Tue', 'Thu']
    >>> list(Crontab.iter_datetimes("0 0 11-19/4,8,30 * *", count=10, start_date=start_date,
    ...     callback=lambda i: i.strftime("%m-%d")))
    ['02-08', '02-11', '02-15', '02-19', '03-08', '03-11', '03-15', '03-19', '03-30', '04-08']
    >>> list(Crontab.iter_datetimes("* * * * *", count=1, start_date=start_date))
    [datetime.datetime(2023, 2, 1, 0, 0)]
    >>> list(Crontab.iter_datetimes("5 4-5,6-9/2 5,6 * 3,5", count=3, start_date=start_date, callback=str))
    ['2023-02-01 04:05:00', '2023-02-01 05:05:00', '2023-02-01 06:05:00']
    >>> list(Crontab.iter_datetimes("0 0 1 8,9,10 *", count=3, start_date=start_date, callback=str, yield_tries=True))
    [(7, '2023-08-01 00:00:00'), (120, '2023-09-01 00:00:00'), (232, '2023-10-01 00:00:00')]
    >>> list(Crontab.iter_datetimes("0 0 1 11 *", count=3, start_date=start_date, callback=str, yield_tries=True))
    [(10, '2023-11-01 00:00:00'), (133, '2024-11-01 00:00:00'), (256, '2025-11-01 00:00:00')]
    >>> list(Crontab.iter_datetimes("*/15   *  * * *", count=2, start_date=start_date, callback=str))
    ['2023-02-01 00:00:00', '2023-02-01 00:15:00']
    >>> list(Crontab.iter_datetimes("0 9 1 * 5", count=5, start_date=start_date, callback=str))
    ['2023-02-01 09:00:00', '2023-02-03 09:00:00', '2023-02-10 09:00:00', '2023-02-17 09:00:00', '2023-02-24 09:00:00']
    >>> list(Crontab.iter_datetimes("abc * * * *", count=1, start_date=start_date))
    Traceback (most recent call last):
        ...
    ValueError: invalid crontab field for minute: 'abc'

    Notes:
        - Fields may be separated by any whitespace, not only a single space.
        - Malformed or out-of-range fields raise ValueError instead of silently
          returning a pattern that never matches.
        - Day-of-month and day-of-week follow Vixie cron: they are ANDed when
          either field is unrestricted ("*"), and ORed when both are restricted.
        - ValueError is raised when no match is found within max_tries, instead
          of silently yielding nothing.
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

    _field_part_re = re.compile(r"(?P<base>\*|\d+|\d+-\d+)(?:/(?P<step>\d+))?")

    @classmethod
    def parse_field(
        cls, value: str, range_list: List[int], field: str = "field"
    ) -> Set[int]:
        """Parse a single cron field into a set of matching values.

        Malformed or out-of-range values raise ValueError instead of being
        silently dropped, which would produce a pattern that never matches.
        """
        if value == "*":
            return set(range_list)
        lo, hi = range_list[0], range_list[-1]
        results: Set[int] = set()
        for part in value.split(","):
            start, end, step = cls._parse_part(part, lo, hi, field)
            results.update(range(start, end + 1, step))
        return results

    @classmethod
    def _parse_part(
        cls, part: str, lo: int, hi: int, field: str
    ) -> Tuple[int, int, int]:
        """Turn a single cron field part into an inclusive (start, end, step)."""
        match = cls._field_part_re.fullmatch(part)
        if not match:
            raise ValueError(f"invalid crontab field for {field}: {part!r}")
        base = match["base"]
        step = int(match["step"] or 1)
        if step < 1:
            raise ValueError(f"invalid crontab step for {field}: {part!r}")
        if base == "*":
            return lo, hi, step
        if "-" in base:
            start_text, _, end_text = base.partition("-")
            start, end = int(start_text), int(end_text)
        else:
            start = end = int(base)
            if step != 1:
                raise ValueError(
                    f"invalid crontab field for {field}: {part!r} "
                    f"(use {base}-{hi}/{step} or */{step})"
                )
        if not (lo <= start <= end <= hi):
            raise ValueError(
                f"crontab field {field} out of range {lo}-{hi} or reversed: {part!r}"
            )
        return start, end, step

    @staticmethod
    def _match_day(
        date: datetime, days: Set[int], py_weekdays: Set[int], dom_dow_or: bool
    ) -> bool:
        """Match day-of-month and day-of-week the way Vixie cron does."""
        day_matched = date.day in days
        weekday_matched = date.weekday() in py_weekdays
        if dom_dow_or:
            return day_matched or weekday_matched
        return day_matched and weekday_matched

    @classmethod
    def _parse_pattern(
        cls, pattern: str
    ) -> Tuple[Set[int], Set[int], Set[int], Set[int], Set[int], bool]:
        """Parse and validate a 5-field crontab pattern."""
        fields = pattern.split()
        if len(fields) != 5:
            raise ValueError(
                f"crontab pattern must have exactly 5 fields, got {len(fields)}: {pattern!r}"
            )
        M, H, d, m, w = fields
        mins = cls.parse_field(M, cls.mins, "minute")
        hours = cls.parse_field(H, cls.hours, "hour")
        days = cls.parse_field(d, cls.days, "day of month")
        months = cls.parse_field(m, cls.months, "month")
        weeks = cls.parse_field(w, cls.weeks, "day of week")
        py_weekdays = {cls.wrap_cron_to_python[i] for i in weeks}
        # Vixie cron: DOM and DOW are ORed only when both are restricted.
        dom_dow_or = days != set(cls.days) and weeks != set(cls.weeks)
        return mins, hours, days, months, py_weekdays, dom_dow_or

    @classmethod
    def _next_candidate(
        cls,
        current: datetime,
        parsed: Tuple[Set[int], Set[int], Set[int], Set[int], Set[int], bool],
    ) -> datetime:
        """Fast forward to the next possibly matching datetime, or return it as is."""
        mins, hours, days, months, py_weekdays, dom_dow_or = parsed
        if current.month not in months:
            # fast forward to next month with performance improvement
            next_month = current.replace(day=28, hour=0, minute=0) + timedelta(days=4)
            return next_month.replace(day=1)
        if not cls._match_day(current, days, py_weekdays, dom_dow_or):
            return current.replace(hour=0, minute=0) + timedelta(days=1)
        if current.hour not in hours:
            return current.replace(minute=0) + timedelta(hours=1)
        if current.minute not in mins:
            return current + timedelta(minutes=1)
        return current

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
            # naive local time is the scheduling contract, so no tzinfo here
            _start_date = datetime.now().replace(microsecond=0)  # noqa: DTZ005
        else:
            _start_date = start_date
        _start_date = _start_date.replace(second=0)
        parsed = cls._parse_pattern(pattern)
        tries = 0
        matched = 0
        last_match = None
        try:
            while tries < max_tries:
                tries += 1
                candidate = cls._next_candidate(_start_date, parsed)
                if candidate != _start_date:
                    _start_date = candidate
                    continue
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
            else:
                raise ValueError(
                    f"no matching datetime within max_tries={max_tries} "
                    f"for crontab pattern: {pattern!r}"
                )
        except OverflowError:
            raise ValueError(
                f"crontab pattern never matches a valid datetime: {pattern!r}"
            ) from None


if __name__ == "__main__":
    import doctest

    doctest.testmod()
