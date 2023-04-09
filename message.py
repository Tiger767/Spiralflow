import collections
import copy
from typing import Any, Callable, Dict, List, Optional, Set
from abc import ABC, abstractmethod
import regex as re
import json


def extract_fstring_variables(text: str) -> List[str]:
    pattern = r"{([a-zA-Z_][\w]*?(?:\.[a-zA-Z_]+)*?)}"
    variable_names = re.findall(pattern, text)
    return variable_names


def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
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
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class ExtractionError(Exception):
    pass

class Role:
    """
    A class to represent the role of a message. Using OpenAI roles.
    """
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'


class Message(ABC):
    def __init__(self, content_format: str, role: Optional[str] = None) -> None:
        """
        :param content_format: A f-string format for the message content.
        :param role: Role associated with the message (default is None).
        """
        self._content_format = content_format
        self._content = None
        self._role = role
        self._variables = {variable: None for variable in extract_fstring_variables(content_format)}
        self._varnames = set(self._variables)
        self._const = False

    @property
    def content_format(self) -> str:
        return self._content_format
    
    @content_format.setter
    def content_format(self, content_format: str):
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
        if self._const:
            raise ValueError('Message is const')
        self._role = role

    @property
    def variables(self) -> Dict[str, Any]:
        return copy.deepcopy(self._variables)
    
    @property
    def varnames(self) -> Set[str]:
        return copy.deepcopy(self._varnames)

    def defined(self) -> bool:
        """
        :return: True if all variables have a value, False otherwise.
        """
        return all(value is not None for value in self._variables.values())

    def make_const(self) -> None:
        self._const = True

    def get_const(self) -> Any:
        """
        :return: A deepcopy of this message made constant so variables and content format cannot change.
        """
        message = copy.deepcopy(self)
        message.make_const()
        return message

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        return super().__call__(**kwargs)
    
    def __str__(self) -> str:
        """
        :return: The message content if defined, otherwise the message content format.
        """
        if self.defined():
            return self.content
        return self.content_format


class InputMessage(Message):
    def __init__(self, content_format: str, role: Optional[str] = Role.USER, custom_insert_variables_func: Optional[Callable[[Dict[str, Any]], str]] = None):
        """
        :param content_format: A f-string format for the message content.
        :param role: Role associated with the message (default is None).
        :param custom_insert_variables_func: A custom function to insert variables into the message content.
                                             Takes the content_format and a dictionary of variables and returns the message content.
        """
        super().__init__(content_format, role)
        self.custom_insert_variables_func = custom_insert_variables_func

    def __call__(self,  **kwargs: Any) -> str:
        """
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
        :param variables: A dictionary containing variable values.
        :return: The message content with inserted variables.
        """
        if self.custom_insert_variables_func:
            return self.custom_insert_variables_func(self._content_format, variables)
        return self._content_format.format(**variables)


class OutputMessage(Message):
    def __init__(self, content_format: str, role: Optional[str] = Role.ASSISTANT, custom_extract_variables_func: Optional[Callable[[List[str], str, str], Dict[str, Any]]] = None):
        """
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
                raise ValueError('Could not extract variables from message content')
            return result.groupdict()


class InputJSONMessage(InputMessage):
    def __init__(self, content_format: str, role: Optional[str] = Role.USER):
        super().__init__(content_format, role, custom_insert_variables_func=self.insert_variables_into_json)
        self._varnames = set(varname.split('.', 1)[0] for varname in self._variables)

    def insert_variables_into_json(self, content_format: str, variables: Dict[str, Any]) -> str:
        """
        :param content_format: The message content format.
        :param variables: A dictionary containing variable values.
        :return: The message content with inserted variables.
        """
        variables = flatten_dict(variables)
        print(content_format)
        content_format = content_format.replace('.', '_') # TEMP FIX!!
        return content_format.format(**variables)


class OutputJSONMessage(OutputMessage):
    def __init__(self, content_format: str, role: Optional[str] = Role.ASSISTANT):
        super().__init__(content_format, role, custom_extract_variables_func=self.extract_variables_from_json)

    def extract_variables_from_json(self, names: List[str], content_format: str, content: str) -> Dict[str, Any]:
        """
        :param names: A list of variable names.
        :param content_format: The message content format.
        :param content: The message content to extract variables from.
        :return: A dictionary containing the extracted variables.
        """
        pattern = re.escape(content_format).replace('\\{', '{').replace('\\}', '}').format(**{name: '(?P<{}>[\s\S]*)'.format(name) for name in names})
        result = re.match(pattern, content)
        if result is None:
            raise ExtractionError('Could not extract variables from JSON message content.')
        variables = result.groupdict()

        json_variables = {}
        try:
            for varname, varvalue in variables.items():
                json_variables[varname] = json.loads(varvalue)
        except json.JSONDecodeError:
            raise ExtractionError('Could not decode variables from JSON message content.')
        return json_variables
