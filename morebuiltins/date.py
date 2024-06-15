import re
from datetime import datetime, timedelta
from typing import List, Optional, Union


class ScheduleTimer:
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
            # many clocks
            if a == time_range:
                return True
            else:
                match = range_regex.match(time_range)
                if match and int(a) >= int(match[1]) and int(a) <= int(match[2]):
                    return True
        return False

    @classmethod
    def get_next_datetimes(
        cls,
        pattern: str,
        count: int = 1,
        from_date: Optional[datetime] = None,
        step: timedelta = timedelta(seconds=1),
        include_from_date: bool = True,
        max_tries: int = 10000,
        strftime: Optional[str] = None,
    ) -> List[Union[datetime, str]]:
        """
        Generates a list of upcoming datetimes that match the given pattern.

        Args:
            pattern (str): The datetime pattern to match against.
            count (int): The number of datetimes to generate.
            from_date (Optional[datetime]): The starting datetime. Defaults to the current datetime.
            step (timedelta): The interval between tries. Defaults to 1 second.
            include_from_date (bool): Whether to consider `from_date` as a potential match.
            max_tries (int): Maximum attempts to find matching datetimes.
            strftime (Optional[str]): Format to apply to matched datetimes if returned as strings.

        Returns:
            List[Union[datetime, str]]: List of matching datetimes or their string representations.
        """
        if from_date is None:
            from_date = datetime.now()
        if include_from_date:
            current = from_date
        else:
            current = from_date + step
        result: List[Union[datetime, str]] = []
        for _ in range(max_tries):
            if cls.is_at(pattern, current):
                if strftime:
                    result.append(current.strftime(strftime))
                else:
                    result.append(current)
                if len(result) == count:
                    break
            current += step
        return result

    @classmethod
    def is_at(cls, pattern: str, date: Optional[datetime] = None):
        """
        Evaluates if the given datetime (or the current datetime if not provided) satisfies the pattern.

        Args:
            pattern (str): The datetime pattern to evaluate.
            date (Optional[datetime]): The datetime to check against the pattern. Defaults to the current datetime.

        Returns:
            bool: True if the datetime matches the pattern, False otherwise.
        """
        if date is None:
            date = datetime.now()
        for sub_pattern in cls.split_or.split(pattern):
            for and_pattern in cls.split_and.split(sub_pattern):
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
                            return True
                    elif result:
                        return True
        return False
