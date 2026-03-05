import inspect
import json
import logging
import re


def get_class(classname):
    """
    Returns the class object associated with the dot-notation classname.

    Taken from here: http://stackoverflow.com/a/452981

    :param classname: the classname
    :type classname: str
    :return: the class object
    :rtype: object
    """
    parts = classname.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


from_dict_handlers = {}
"""
The methods that handle the restoration from a JSON dictionary, stored under their 'type'.
"""


def register_dict_handler(typestr, handler):
    """
    Registers a handler for restoring an object from a JSON dictionary.

    :param typestr: the type of the object
    :type typestr: str
    :param handler: the method
    """
    global from_dict_handlers
    from_dict_handlers[typestr] = handler


def deregister_dict_handler(typestr):
    """
    Deregisters a handler for restoring an object from a JSON dictionary.

    :param typestr: the type of the object
    :type typestr: str
    """
    global from_dict_handlers
    del from_dict_handlers[typestr]


def has_dict_handler(typestr):
    """
    Returns the handler for restoring an object from a JSON dictionary.

    :param typestr: the type of the object
    :type typestr: str
    :return: the handler, None if not available
    """
    global from_dict_handlers
    return typestr in from_dict_handlers


def get_dict_handler(typestr):
    """
    Returns the handler for restoring an object from a JSON dictionary.

    :param typestr: the type of the object
    :type typestr: str
    :return: the handler, None if not available
    """
    global from_dict_handlers
    return from_dict_handlers[str(typestr)]


class JSONObject(object):
    """
    Ancestor for classes that can be represented as JSON and restored from JSON.
    """

    def to_dict(self):
        """
        Returns a dictionary that represents this object, to be used for JSONification.

        :return: the object dictionary
        :rtype: dict
        """
        raise Exception("Not implemented!")

    @classmethod
    def from_dict(cls, d):
        """
        Restores an object state from a dictionary, used in de-JSONification.

        :param d: the object dictionary
        :type d: dict
        :return: the object
        :rtype: object
        """
        raise Exception("Not implemented!")

    def to_json(self):
        """
        Returns the options as JSON.

        :return: the object as string
        :rtype: str
        """
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, separators=(',', ': '))

    @classmethod
    def from_json(cls, s):
        """
        Restores the object from the given JSON.

        :param s: the JSON string to parse
        :type s: str
        :return: the
        """
        d = json.loads(s)
        return get_dict_handler(d["type"])(d)

    def shallow_copy(self):
        """
        Returns a shallow copy of itself.

        :return: the copy
        :rtype: object
        """
        return self.from_json(self.to_json())


class Configurable(JSONObject):
    """
    The ancestor for all actors.
    """

    def __init__(self, config=None):
        """
        Initializes the object.

        :param config: the dictionary with the options (str -> object).
        :type config: dict
        """
        self._logger = None
        self._help = {}
        self._config = self.fix_config({})
        if config is not None:
            self.config = config
        if not has_dict_handler("Configurable"):
            register_dict_handler("Configurable", Configurable.from_dict)

    def __repr__(self):
        """
        Returns Python code for instantiating the object.

        :return: the representation
        :rtype: str
        """
        return \
            self.__class__.__module__ + "." + self.__class__.__name__ \
            + "(config=" + str(self.config) + ")"

    def description(self):
        """
        Returns a description of the object.

        :return: the description
        :rtype: str
        """
        raise Exception("Not implemented!")

    def fix_config(self, options):
        """
        Fixes the options, if necessary. I.e., it adds all required elements to the dictionary.

        :param options: the options to fix
        :type options: dict
        :return: the (potentially) fixed options
        :rtype: dict
        """
        return options

    @property
    def config(self):
        """
        Obtains the currently set options of the actor.

        :return: the options
        :rtype: dict
        """
        return self._config

    @config.setter
    def config(self, options):
        """
        Sets the options of the actor.

        :param options: the options
        :type options: dict
        """
        self._config = self.fix_config(options)

    def get_classname(self, obj):
        """
        Returns the classname of the JB_Object, Python class or object.

        :param obj: the java object or Python class/object to get the classname for
        :type obj: object
        :return: the classname
        :rtype: str
        """
        if inspect.isclass(obj):
            return obj.__module__ + "." + obj.__name__
        else:
            return self.get_classname(obj.__class__)

    def to_dict(self):
        """
        Returns a dictionary that represents this object, to be used for JSONification.

        :return: the object dictionary
        :rtype: dict
        """
        result = dict()
        result["type"] = "Configurable"
        result["class"] = self.get_classname(self)
        result["config"] = {}
        for k in self._config:
            v = self._config[k]
            if isinstance(v, JSONObject):
                result["config"][k] = v.to_dict()
            else:
                result["config"][k] = v
        return result

    @classmethod
    def from_dict(cls, d):
        """
        Restores its state from a dictionary, used in de-JSONification.

        :param d: the object dictionary
        :type d: dict
        """
        conf = {}
        for k in d["config"]:
            v = d["config"][k]
            if isinstance(v, dict):
                conf[str(k)] = get_dict_handler(d["config"]["type"])(v)
            else:
                conf[str(k)] = v
        return get_class(str(d["class"]))(config=conf)

    def new_logger(self):
        """
        Returns a new logger instance.

        :return: the logger instance
        :rtype: logger
        """
        return logging.getLogger(self.get_classname(self))

    @property
    def logger(self):
        """
        Returns the logger object.

        :return: the logger
        :rtype: logger
        """
        if self._logger is None:
            self._logger = self.new_logger()
        return self._logger

    @property
    def help(self):
        """
        Obtains the help information per option for this actor.

        :return: the help
        :rtype: dict
        """
        return self._help

    def generate_help(self):
        """
        Generates a help string for this actor.

        :return: the help string
        :rtype: str
        """
        result = list()
        result.append(self.__class__.__name__)
        result.append(re.sub(r'.', '=', self.__class__.__name__))
        result.append("")
        result.append("DESCRIPTION")
        result.append(self.description())
        result.append("")
        result.append("OPTIONS")
        opts = sorted(self.config.keys())
        for opt in opts:
            result.append(opt)
            helpstr = self.help[opt]
            if helpstr is None:
                helpstr = "-missing help-"
            result.append("\t" + helpstr)
            result.append("")
        return '\n'.join(result)

    def print_help(self):
        """
        Prints a help string for this actor to stdout.
        """
        print(self.generate_help())
