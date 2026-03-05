FIRST = "first"
SECOND = "second"
THIRD = "third"
LAST = "last"
LAST_1 = "last_1"
LAST_2 = "last_2"
ALL = FIRST + "-" + LAST
PLACEHOLDERS = [
    FIRST,
    SECOND,
    THIRD,
    LAST_2,
    LAST_1,
    LAST,
]


def is_int(s):
    """
    Checks whether the string represents an integer.

    :param s: the string to check
    :type s: str
    :return: true if an integer
    :rtype: bool
    """
    try:
        int(s)
        return True
    except:
        return False


def is_placeholder(s):
    """
    Checks whether the string represents a valid placeholder (first, second, ...).

    :param s: the string to check
    :type s: str
    :return: whether the string is a valid placeholder
    :rtype: bool
    """
    if s == FIRST:
        return True
    elif s == LAST:
        return True
    elif s == SECOND:
        return True
    elif s == LAST_1:
        return True
    elif s == THIRD:
        return True
    elif s == LAST_2:
        return True
    else:
        return False
