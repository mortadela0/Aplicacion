from simple_range._core import is_int, is_placeholder
from simple_range._core import FIRST, SECOND, THIRD, LAST_2, LAST_1, LAST, PLACEHOLDERS


class Index(object):
    """
    Defines an index object. Uses "human" 1-based indices.
    E.g.: first, last, 1, 12
    """

    def __init__(self, index, maximum=-1):
        """
        Initializes the index.

        :param index: the index definition string
        :type index: str
        :param maximum: the maximum number of elements (incl), -1 for undefined
        :type maximum: int
        """
        self._check(index)
        self._index = index
        self._maximum = maximum
        self._zero_based = None
        self._one_based = None
        self._reset()

    def _reset(self):
        """
        Resets the index cache.
        """
        self._zero_based = None
        self._one_based = None

    def _check(self, s):
        """
        Checks the index definition.
        """
        if not is_int(s) and not is_placeholder(s):
            raise Exception("Invalid index '%s'!" % s)

    @property
    def index(self):
        """
        Returns the index definition.

        :return: the index
        :rtype: str
        """
        return self._index

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

    def value(self, zero_based=True):
        """
        Returns the index as specified.

        :param zero_based: whether to return 0-based or 1-based index
        :type zero_based: bool
        :return: the index
        :rtype: int
        """
        if self._maximum == -1:
            raise Exception("No maximum number of elements specified!")

        # not cached yet?
        if self._zero_based is None:
            if self._index == FIRST:
                self._zero_based = 0
            elif self._index == SECOND:
                self._zero_based = 1
            elif self._index == THIRD:
                self._zero_based = 2
            elif self._index == LAST_2:
                self._zero_based = self._maximum - 3
            elif self._index == LAST_1:
                self._zero_based = self._maximum - 2
            elif self._index == LAST:
                self._zero_based = self._maximum - 1
            else:
                self._zero_based = int(self._index) - 1
            self._one_based = self._zero_based + 1

        if zero_based:
            return self._zero_based
        else:
            return self._one_based

    def __str__(self):
        """
        Returns the index definition.

        :return: the index definition
        :rtype: str
        """
        return self._index

    @classmethod
    def help(cls):
        """
        Returns a short help string.

        :return: the help string
        :rtype: str
        """
        return "1-based index; available placeholders: " + ", ".join(PLACEHOLDERS) + "; " \
               + LAST_1 + ": second to last, " + LAST_2 + ": third to last"


def index_value(s, maximum, zero_based=True):
    """
    Convenience method to convert an index string into an integer.

    :param s: the index string to parse/convert
    :type s: str
    :param maximum: the maximum number of elements
    :type maximum: int
    :param zero_based: whether to return 0-based or 1-based index
    :type zero_based: bool
    :return: the index
    :rtype: int
    """
    i = Index(s, maximum=maximum)
    return i.value(zero_based=zero_based)


if __name__ == "__main__":
    r = Index("1", maximum=10)
    print(r, "-->", r.value())
    r = Index("second", maximum=10)
    print(r, "-->", r.value(zero_based=False))
    r = Index("last", maximum=40)
    print(r, "-->", r.value())
    print(r, "-->", r.value(zero_based=False))
    print(index_value("last_1", maximum=100, zero_based=False))
