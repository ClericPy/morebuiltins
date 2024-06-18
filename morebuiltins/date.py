import re
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple, Union


# NotImplemented yet
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
        Negation (!): Inverts the truth of the following condition.
            Example: "!hour=12" excludes noon.

    Typical Pattern Examples:

        Daily at Midnight
            Pattern: "hour=00&minute=00&second=00"
        Weekdays Except Fridays, Every 3 Hours Starting at 9 AM
            Pattern: "weekday>=MO&weekday<=TH&hour>=9;hour=9+3/3"
        First Monday of Every Month at 10 AM
            Pattern: "day=1&weekday=MO&hour=10&minute=00"
        Every 15 Minutes During Business Hours
            Pattern: "hour>=9&hour<=17&minute/15"
        Alternating Fridays, 2 PM to 4 PM
            Pattern: "weekday=FRI&hour>=14&hour<=16&!(weekday@2,4,6,8,10,12)"
    """

    prefix_not = "!"
    split_or = re.compile(";|\|")
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

    def __init__(self):
        raise NotImplementedError("Performance issue needs to be optimized")

    @staticmethod
    def divide(a: str, b: str):
        if a.isdigit() and b.isdigit():
            return int(a) % int(b) == 0
        else:
            return False

    @staticmethod
    def between(a: str, b: str):
        # many ranges
        range_regex = re.compile("^(\d+)-(\d+)$")
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
                return delta
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
    def get_next_datetimes(
        cls,
        pattern: str,
        count: int = 1,
        start_date: Optional[datetime] = None,
        max_tries: int = 1000000,
        callback: Callable = lambda x: x,
    ) -> List[Union[datetime, str]]:
        if start_date is None:
            start_date = datetime.now().replace(microsecond=0)
        result: List[Union[datetime, str]] = []
        pattern_list, steps = cls.sort_pattern(pattern)
        min_step = min(steps.values())
        min_step = timedelta(seconds=1)
        for _ in range(max_tries):
            ok, miss_pattern = cls.is_at(pattern_list, start_date)
            if ok:
                item = callback(start_date)
                if not result or item != result[-1]:
                    result.append(item)
                    if len(result) == count:
                        break
                step = min_step
            else:
                step = steps[miss_pattern]
            start_date += step
        return result

    @classmethod
    def is_at(
        cls, pattern: Union[list, str], date: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        if date is None:
            date = datetime.now()
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
