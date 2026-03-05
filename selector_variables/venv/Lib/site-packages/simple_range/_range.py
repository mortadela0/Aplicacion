from simple_range._core import is_int, is_placeholder
from simple_range._core import FIRST, SECOND, THIRD, LAST_2, LAST_1, LAST, PLACEHOLDERS


class SubRange(object):
    """
    Defines a subrange, with LOWER and optional UPPER (both included).
    """

    def __init__(self, lower, upper=None):
        """
        Initializes the sub-range.

        :param lower: the lower bound (int or str)
        :param upper: the upper bound (int or str)
        """
        if is_int(lower):
            lower = int(lower) - 1
        if lower == FIRST:
            self.lower = 0
        elif lower == SECOND:
            self.lower = 1
        elif lower == THIRD:
            self.lower = 2
        else:
            self.lower = lower
        if is_int(upper):
            upper = int(upper) - 1
        self.upper = upper

    def _expand(self, bound, maximum, zero_based=True):
        """
        Applies the maximum number of elements and returns the expanded bound.

        :param bound: the bound to expand (str/int)
        :param maximum: the total number of elements
        :type maximum: int
        :param zero_based: whether to return 0-based or 1-based indices
        :type zero_based: bool
        :return: the expanded bound
        :rtype: int
        """
        if isinstance(bound, int):
            result = bound
        elif bound == LAST:
            result = maximum - 1
        elif bound == LAST_1:
            result = maximum - 2
        elif bound == LAST_2:
            result = maximum - 3
        else:
            raise Exception("Invalid bound: %s" + str(bound))
        if not zero_based:
            result += 1
        return result

    def apply(self, maximum, zero_based=True):
        """
        Applies the maximum number of elements and returns the expanded
        list of indices defined by this sub-range.

        :param maximum: the total number of elements
        :type maximum: int
        :param zero_based: whether to return 0-based or 1-based indices
        :type zero_based: bool
        :return: the list of indices that this sub-range represents
        :rtype: list
        """
        result = []
        lower = self._expand(self.lower, maximum, zero_based=zero_based)
        if self.upper is not None:
            upper = self._expand(self.upper, maximum, zero_based=zero_based)
        else:
            upper = None
        # lower < upper?
        if (upper is not None) and (upper == lower):
            upper = None
        if (upper is not None) and (upper < lower):
            raise Exception("Upper bound (%s) < lower bound (%s)!" % (self.upper+1, self.lower+1))
        if upper is not None:
            for i in range(lower, upper + 1):
                result.append(i)
        else:
            result.append(lower)
        return result


class Range(object):
    """
    Defines a range object, a more flexible slicing. Uses "human" 1-based indices.
    E.g.: first-last, 1-10,12,20-last
    """

    def __init__(self, range, maximum=-1):
        """
        Initializes the range.

        :param range: the range definition string
        :type range: str
        :param maximum: the maximum number of elements (incl)
        :type maximum: int
        """
        self._range = range.replace(" ", "")
        self._maximum = maximum
        self._subranges = None
        self._zero_based = None
        self._one_based = None
        self._reset()
        self._parse()

    def _reset(self):
        """
        Resets the index cache.
        """
        self._subranges = None
        self._zero_based = None
        self._one_based = None

    def _parse(self):
        """
        Parses the range definition.
        """
        subranges = []
        parts = self._range.split(",")
        for part in parts:
            if "-" in part:
                subparts = part.split("-")
                if len(subparts) != 2:
                    raise Exception("Invalid sub-range '%s' in range '%s'!" % (part, self._range))
                if not is_int(subparts[0]) and not is_placeholder(subparts[0]):
                    raise Exception("Lower bound '%s' is neither integer nor placeholder in range '%s'!" % (subparts[0], self._range))
                if not is_int(subparts[1]) and not is_placeholder(subparts[1]):
                    raise Exception("Upper bound '%s' is neither integer nor placeholder in range '%s'!" % (subparts[1], self._range))
                subranges.append(SubRange(subparts[0], upper=subparts[1]))
            else:
                if not is_int(part) and not is_placeholder(part):
                    raise Exception("Invalid index '%s' in range '%s'!" % (part, self._range))
                subranges.append(SubRange(part))
        self._subranges = subranges

    @property
    def range(self):
        """
        Returns the range definition.

        :return: the range
        :rtype: str
        """
        return self._range

    @property
    def maximum(self):
        """
        Returns the maximum number of elements (incl).

        :return: the maximum number
        :rtype: int
        """
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        """
        Sets the maximum number of elements (incl).

        :param maximum: the maximum number
        :type maximum: int
        """
        self._maximum = maximum
        self._reset()

    def indices(self, zero_based=True):
        """
        Returns the indices specified

        :param zero_based: whether to return 0-based or 1-based indices
        :type zero_based: bool
        :return: the indices
        :rtype: list
        """
        if self._maximum == -1:
            raise Exception("No maximum number of elements specified!")

        # not cached yet?
        if self._zero_based is None:
            self._parse()
            self._zero_based = []
            self._one_based = []
            for subrange in self._subranges:
                self._zero_based.extend(subrange.apply(self.maximum, zero_based=True))
                self._one_based.extend(subrange.apply(self.maximum, zero_based=False))

        if zero_based:
            return self._zero_based
        else:
            return self._one_based

    def __str__(self):
        """
        Returns the range definition.

        :return: the range definition
        :rtype: str
        """
        return self._range

    @classmethod
    def help(cls):
        """
        Returns a short help string.

        :return: the help string
        :rtype: str
        """
        return "1-based range, e.g., 'first-last' or '1-12,20,22,25-last'; " \
               + "available placeholders: " + ", ".join(PLACEHOLDERS) + ";" \
               + LAST_1 + ": second to last, " + LAST_2 + ": third to last"


def range_indices(s, maximum, zero_based=True):
    """
    Convenience method to convert a range string into a list of indices.
    
    :param s: the range string to parse/convert
    :type s: str
    :param maximum: the maximum number of elements
    :type maximum: int
    :param zero_based: whether to return 0-based or 1-based indices
    :type zero_based: bool
    :return: the list of indices
    :rtype: list
    """
    r = Range(s, maximum=maximum)
    return r.indices(zero_based=zero_based)


if __name__ == "__main__":
    r = Range("1-3", maximum=10)
    print(r, "-->", r.indices())
    r = Range("first-last", maximum=10)
    print(r, "-->", r.indices(zero_based=False))
    r = Range("first-3,11,14-14,30,last", maximum=40)
    print(r, "-->", r.indices())
    print(r, "-->", r.indices(zero_based=False))
    print(range_indices("2-last_2", maximum=10))
