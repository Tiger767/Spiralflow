import collections
import copy
from typing import Any, Callable, Dict, List, Optional, Set
from abc import ABC, abstractmethod
import regex as re
import json


def extract_fstring_variables(text: str) -> List[str]:
    """
    Extracts variables from a f-string like text.

    :param text: f-string like text to extract variables from.
    """
    pattern = r"(?<!{){([a-zA-Z_][\w]*?(?:\.[a-zA-Z_]+)*?)}(?!})"
    variable_names = re.findall(pattern, text)
    return variable_names


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Flatten a dictionary.

    :param d: Dictionary to flatten.
    :param parent_key: Parent key to use.
    :param sep: Separator to use.
    :return: Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class ExtractionError(Exception):
    """
    A class to represent an error in extracting a variable from a message.
    """
    pass


class Role:
    """
    A class to represent the role of a message. Using OpenAI roles.
    """
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'


class Message(ABC):
    """
    A class to represent a message.
    """
    
    def __init__(self, content_format: str, role: Optional[str] = None) -> None:
        """
        Initializes the Message class with the given parameters.

        :param content_format: A f-string format for the message content.
        :param role: Role associated with the message (default is None).
        """
        self._content_format = content_format
        self._content = None
        self._role = role
        self._variables = {variable: None for variable in extract_fstring_variables(content_format)}
        self._varnames = set(self._variables)
        self._const = False

        if len(self._variables) == 0:
            self._content = content_format

    @property
    def content_format(self) -> str:
        return self._content_format
    
    @content_format.setter
    def content_format(self, content_format: str):
        """
        :param content_format: A f-string like format for the message content.
        """
        if self._const:
            raise ValueError('Message is const')
        self._content_format = content_format
        self._variables = {variable: None for variable in extract_fstring_variables(content_format)}
        self._varnames = set(self._variables)

    @property
    def content(self) -> str:
        return self._content

    @property
    def role(self) -> str:
        return self._role

    @role.setter
    def role(self, role: str):
        """
        :param role: Role associated with the message.
        """
        if self._const:
            raise ValueError('Message is const')
        self._role = role

    @property
    def variables(self) -> Dict[str, Any]:
        return copy.deepcopy(self._variables)
    
    @property
    def varnames(self) -> Set[str]:
        return set(self._varnames)

    def defined(self) -> bool:
        """
        Determines if all variables have a value, essentially if the message has been called or has no variables.

        :return: True if all variables have a value, False otherwise.
        """
        return all(value is not None for value in self._variables.values())

    def make_const(self) -> None:
        """
        Makes this message constant so variables and content format cannot change.
        """
        self._const = True

    def get_const(self) -> Any:
        """
        Creates a deepcopy of self and makes it constant.

        :return: A deepcopy of this message made constant so variables and content format cannot change.
        """
        message = copy.deepcopy(self)
        message.make_const()
        return message

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        """
        A method to run content through to get variables or to put variables in to form a content.
        """
    
    def __str__(self) -> str:
        """
        :return: The message content if defined, otherwise the message content format.
        """
        if self.defined():
            return self.content
        return self.content_format


class InputMessage(Message):
    """
    A class to represent a message that takes variables as inputs to construct.
    """

    def __init__(self, content_format: str, role: Optional[str] = Role.USER, custom_insert_variables_func: Optional[Callable[[Dict[str, Any]], str]] = None):
        """
        Initializes the InputMessage class with the given parameters.

        :param content_format: A f-string format for the message content.
        :param role: Role associated with the message (default is None).
        :param custom_insert_variables_func: A custom function to insert variables into the message content.
                                             Takes the content_format and a dictionary of variables and returns the message content.
        """
        super().__init__(content_format, role)
        self.custom_insert_variables_func = custom_insert_variables_func

    def __call__(self,  **kwargs: Any) -> str:
        """
        Get the message content with inserted variables.

        :param kwargs: A dictionary containing variable values.
        :return: The message content with inserted variables.
        """
        if self._const:
            return self.insert_variables(kwargs)
        self._variables = {varname: varvalue for varname, varvalue in kwargs.items() if varname in self._varnames}
        self._content = self.insert_variables(self._variables)
        return self.content

    def insert_variables(self, variables: Dict[str, Any]) -> str:
        """
        Insert variables into the message content.

        :param variables: A dictionary containing variable values.
        :return: The message content with inserted variables.
        """
        if self.custom_insert_variables_func:
            return self.custom_insert_variables_func(self._content_format, variables)
        return self._content_format.format(**variables)


class OutputMessage(Message):
    """
    A class to represent a message that outputs variables from its message content.

    Limitations:
    - Variables must be seperated. Regex pattern used: (?P<{}>[\s\S]*)
    """

    def __init__(self, content_format: str, role: Optional[str] = Role.ASSISTANT, custom_extract_variables_func: Optional[Callable[[List[str], str, str], Dict[str, Any]]] = None):
        """
        Initializes the OutputMessage class with the given parameters.

        :param content_format: A f-string format for the message content.
        :param role: Role associated with the message (default is None).
        :param custom_extract_variables_func: A custom function to extract variables from the message content.
                                              Takes a list of variable names, the content format, and the message content.
                                              Returns a dictionary containing the extracted variables.
        """
        super().__init__(content_format, role)
        self.custom_extract_variables_func = custom_extract_variables_func

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Extract variables from the message content.

        :param kwargs: A dictionary containing the message content.
        :return: A dictionary containing the extracted variables.
        """
        if self._const:
            return self.extract_variables(kwargs['content'])
        self._content = kwargs['content']
        self._variables = self.extract_variables(self._content)
        return self.variables

    def extract_variables(self, content) -> Dict[str, Any]:
        """
        Extract variables from the message content.

        :param content: The message content to extract variables from.
        :return: A dictionary containing the extracted variables.
        """
        names = list(self._varnames)
        if self.custom_extract_variables_func:
            return self.custom_extract_variables_func(names, self._content_format, content)
        else:
            pattern = re.escape(self._content_format).replace('\\{', '{').replace('\\}', '}').format(**{name: '(?P<{}>[\s\S]*)'.format(name) for name in names})
            result = re.match(pattern, content)
            if result is None:
                raise ValueError(f'Could not extract variables from message content.\nContent Format: {self._content_format}\nPattern: {pattern}\nContent: {content}')
            return result.groupdict()


class InputJSONMessage(InputMessage):
    """
    A class to represent a message that takes JSON dict keys-values as inputs to construct.

    Limitations:
    - Sub-dictionaries are accessed by periods and replaced with underscores in processing, so name conflicts can occur.
    """

    def __init__(self, content_format: str, role: Optional[str] = Role.USER, expected_input_varnames: Optional[Set[str]] = None):
        """
        Initializes the InputJSONMessage class with the given parameters.

        :param content_format: A f-string format for the message content.
        :param role: Role associated with the message (default is None).
        :param expected_input_varnames: A set of expected input variable names.
        """
        super().__init__(content_format, role, custom_insert_variables_func=self.insert_variables_into_json)
        self._expected_input_varnames = expected_input_varnames
        if len(self._varnames) == 0 and self._expected_input_varnames is not None:
            raise ValueError(f'No variables found in content format but given expected inputs. Expected: {self._expected_input_varnames}')
        else:
            self._varnames = set(varname.split('.', 1)[0] for varname in self._variables)

    def insert_variables_into_json(self, content_format: str, variables: Dict[str, Any]) -> str:
        """
        Insert variables from dict into the message content.

        :param content_format: The message content format.
        :param variables: A dictionary containing variable values.
        :return: The message content with inserted variables.
        """
        variables = flatten_dict(variables)
        if self._expected_input_varnames is not None and not self._expected_input_varnames.issubset(variables.keys()):
            raise ExtractionError(f'Missing expected input variables. Expected: {self._expected_input_varnames}, Actual: {variables.keys()}')
        
        variables = {varname.replace('.', '_'): varvalue for varname, varvalue in variables.items()}
        
        replacement = lambda match: match.group(0).replace('.', '_')
        content_format = re.sub(r"(?<!{){([a-zA-Z_][\w]*?(?:\.[a-zA-Z_]+)*?)}(?!})", replacement, content_format)
        return content_format.format(**variables)


class OutputJSONMessage(OutputMessage):
    """
    A class to represent a message that outputs JSON dict keys-values from its message content.

    Limitations:
    - Only supports JSON dicts as outputs.
    - Regex patterns do not necessarily match every content_format possible.
    """

    def __init__(self, content_format: str, role: Optional[str] = Role.ASSISTANT):
        """
        Initializes the OutputJSONMessage class with the given parameters.

        :param content_format: A f-string format for the message content.
        :param role: Role associated with the message (default is None).
        """
        super().__init__(content_format, role, custom_extract_variables_func=self.extract_variables_from_json)

    def extract_variables_from_json(self, names: List[str], content_format: str, content: str) -> Dict[str, Any]:
        """
        Extract JSON Dict from the message content.

        :param names: A list of variable names.
        :param content_format: The message content format.
        :param content: The message content to extract variables from.
        :return: A dictionary containing the extracted variables.
        """
        pattern = re.escape(content_format).replace('\\{', '{').replace('\\}', '}').format(**{name: '(?P<{}>[\s\S]*)'.format(name) for name in names})
        result = re.match(pattern, content)
        if result is None:
            raise ExtractionError(f'Could not extract variables from JSON message content.\nContent format: {content_format}\nPattern: {pattern}\nContent: {content}')
        variables = result.groupdict()

        json_variables = {}
        try:
            for varname, varvalue in variables.items():
                if len(varvalue.strip()) == 0:
                    json_variables[varname] = {}
                else:
                    json_variables[varname] = json.loads(varvalue)
        except json.JSONDecodeError:
            raise ExtractionError(f'Could not decode variables from JSON message content.\n\n{varname}:\n{varvalue}')
        return json_variables
