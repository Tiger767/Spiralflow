from typing import Any, Callable, Dict, List, Optional, Set
from abc import ABC, abstractmethod
import copy
import regex as re


def extract_fstring_variables(text: str):
    # Regex pattern to match expressions within curly braces
    pattern = r"{(.*?)}"
    matches = re.findall(pattern, text)

    # Extract variable names from the expressions
    variable_names = []
    for match in matches:
        # Remove any whitespace and split by any non-alphanumeric characters
        var_names = re.split(r'\W+', match.strip())
        
        # Add any valid variable names to the list
        for var_name in var_names:
            if var_name.isidentifier():
                variable_names.append(var_name)

    return variable_names


class Role:
    """
    A class to represent the role of a message. Using OpenAI roles.
    """
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'


class Message(ABC):
    def __init__(self, content_format: str, role: Optional[str] = None):
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
    def role(self, role: str) -> None:
        if self._const:
            raise ValueError('Message is const')
        self._role = role

    @property
    def variables(self) -> Dict[str, str]:
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
        if self.defined():
            return self.content
        return self.content_format


class InputMessage(Message):
    def __init__(self, content_format: str, role: Optional[str] = Role.USER):
        """
        :param content_format: A f-string format for the message content.
        :param role: Role associated with the message (default is None).
        """
        super().__init__(content_format, role)

    def __call__(self,  **kwargs: Any) -> Any:
        """
        :param kwargs: A dictionary containing variable values.
        :return: The message content with inserted variables.
        """
        if self._const:
            return self.insert_variables(kwargs)
        self._variables = {varname: varvalue for varname, varvalue in kwargs.items() if varname in self._varnames}
        self._content = self.insert_variables(self._variables)
        return self.content

    def insert_variables(self, variables) -> str:
        """
        :param variables: A dictionary containing variable values.
        :return: The message content with inserted variables.
        """
        return self._content_format.format(**variables)


class OutputMessage(Message):
    def __init__(self, content_format: str, role: Optional[str] = Role.ASSISTANT, custom_extract_variables_func: Optional[Callable[[List[str], str, str], Dict[str, str]]] = None):
        """
        :param content_format: A f-string format for the message content.
        :param role: Role associated with the message (default is None).
        :param custom_extract_variables_func: A custom function to extract variables from the message content.
                                              Takes a list of variable names, the content format, and the message content.
                                              Returns a dictionary containing the extracted variables.
        """
        super().__init__(content_format, role)
        self.custom_extract_variables_func = custom_extract_variables_func

    def __call__(self, **kwargs: Any) -> Any:
        """
        :param kwargs: A dictionary containing the message content.
        :return: A dictionary containing the extracted variables.
        """
        if self._const:
            return self.extract_variables(kwargs['content'])
        self._content = kwargs['content']
        self._variables = self.extract_variables(self._content)
        return self.variables

    def extract_variables(self, content) -> Dict[str, str]:
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
                return None
            return result.groupdict()
