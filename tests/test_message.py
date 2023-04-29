import pytest
from spiralflow.message import (
    extract_fstring_variables,
    flatten_dict,
    ExtractionError,
    Role,
    Message,
)
import pytest
from typing import Dict, Any, Optional, Callable
from spiralflow.message import InputMessage, Message, Role
import pytest
import re
from typing import Dict, Any, Optional, Callable, List
from spiralflow.message import OutputMessage, Message, Role, ExtractionError
import pytest
import re
from typing import Dict, Any, Optional, Callable, List
from spiralflow.message import OutputMessage, Message, Role, ExtractionError
import json
import pytest
from spiralflow.message import (
    InputJSONMessage,
    OutputJSONMessage,
    Role,
    ExtractionError,
    flatten_dict,
)
import pytest
from spiralflow.message import OutputOptions, OutputMessage, Role, ExtractionError


def test_extract_fstring_variables():
    test_text = "Hello, {name}. Your age is {age}."
    expected_result = ["name", "age"]
    assert extract_fstring_variables(test_text) == expected_result


def test_flatten_dict():
    test_dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    expected_result = {"a": 1, "b.c": 2, "b.d.e": 3}
    assert flatten_dict(test_dict) == expected_result


class TestInputMessage:
    def test_init(self):
        msg = InputMessage("{greeting} {name}!", Role.ASSISTANT)
        assert isinstance(msg, InputMessage)
        assert isinstance(msg, Message)
        assert msg._content_format == "{greeting} {name}!"
        assert msg.role == Role.ASSISTANT
        assert msg.custom_insert_variables_func is None

    def test_call(self):
        msg = InputMessage("{greeting} {name}!", Role.ASSISTANT)
        content = msg(greeting="Hello", name="John")
        assert content == "Hello John!"

    def test_insert_variables_default(self):
        msg = InputMessage("{greeting} {name}!", Role.ASSISTANT)
        content = msg.insert_variables({"greeting": "Hello", "name": "John"})
        assert content == "Hello John!"

    def test_insert_variables_custom(self):
        def custom_insert_func(content_format: str, variables: Dict[str, Any]) -> str:
            return content_format.replace("{greeting}", variables["greeting"]).replace(
                "{name}", variables["name"]
            )

        msg = InputMessage("{greeting} {name}!", Role.ASSISTANT, custom_insert_func)
        content = msg.insert_variables({"greeting": "Hello", "name": "John"})
        assert content == "Hello John!"

    def test_no_role_provided(self):
        msg = InputMessage("{greeting} {name}!")
        assert msg.role == Role.USER

    def test_missing_variables(self):
        msg = InputMessage("{greeting} {name}!", Role.ASSISTANT)
        with pytest.raises(KeyError):
            msg(greeting="Hello")


@pytest.fixture
def input_message():
    return InputMessage("{greeting} {name}!", Role.ASSISTANT)


def test_init_fixture(input_message):
    assert isinstance(input_message, InputMessage)
    assert isinstance(input_message, Message)
    assert input_message._content_format == "{greeting} {name}!"
    assert input_message.role == Role.ASSISTANT
    assert input_message.custom_insert_variables_func is None


def test_call_fixture(input_message):
    content = input_message(greeting="Hello", name="John")
    assert content == "Hello John!"


def test_insert_variables_default_fixture(input_message):
    content = input_message.insert_variables({"greeting": "Hello", "name": "John"})
    assert content == "Hello John!"


def test_insert_variables_custom_fixture(input_message):
    def custom_insert_func(content_format: str, variables: Dict[str, Any]) -> str:
        return content_format.replace("{greeting}", variables["greeting"]).replace(
            "{name}", variables["name"]
        )

    input_message.custom_insert_variables_func = custom_insert_func
    content = input_message.insert_variables({"greeting": "Hello", "name": "John"})
    assert content == "Hello John!"


def test_no_role_provided_fixture():
    msg = InputMessage("{greeting} {name}!")
    assert msg.role == Role.USER


def test_missing_variables_fixture(input_message):
    with pytest.raises(KeyError):
        input_message(greeting="Hello")


class TestOutputMessage:
    def test_init(self):
        msg = OutputMessage("{greeting} {name}!", Role.ASSISTANT)
        assert isinstance(msg, OutputMessage)
        assert isinstance(msg, Message)
        assert msg._content_format == "{greeting} {name}!"
        assert msg.role == Role.ASSISTANT
        assert msg.custom_extract_variables_func is None

    def test_call(self):
        msg = OutputMessage("{greeting} {name}!", Role.ASSISTANT)
        variables = msg(content="Hello John!")
        assert variables == {"greeting": "Hello", "name": "John"}

    def test_extract_variables_default(self):
        msg = OutputMessage("{greeting} {name}!", Role.ASSISTANT)
        variables = msg.extract_variables("Hello John!")
        assert variables == {"greeting": "Hello", "name": "John"}

    def test_extract_variables_custom(self):
        def custom_extract_func(
            names: List[str], content_format: str, content: str
        ) -> Dict[str, Any]:
            regex_pattern = re.compile(r"^(?P<greeting>\w+) (?P<name>\w+)!$")
            match = regex_pattern.match(content)
            return match.groupdict() if match else None

        msg = OutputMessage("{greeting} {name}!", Role.ASSISTANT, custom_extract_func)
        variables = msg.extract_variables("Hello John!")
        assert variables == {"greeting": "Hello", "name": "John"}

    def test_no_role_provided(self):
        msg = OutputMessage("{greeting} {name}!")
        assert msg.role == Role.ASSISTANT

    def test_extraction_error(self):
        msg = OutputMessage("{greeting} {name}!", Role.ASSISTANT)
        with pytest.raises(ExtractionError):
            msg.extract_variables("Hello")


@pytest.fixture
def output_message():
    return OutputMessage("{greeting} {name}!", Role.ASSISTANT)


def test_init_fixture(output_message):
    assert isinstance(output_message, OutputMessage)
    assert isinstance(output_message, Message)
    assert output_message._content_format == "{greeting} {name}!"
    assert output_message.role == Role.ASSISTANT
    assert output_message.custom_extract_variables_func is None


def test_call_fixture(output_message):
    variables = output_message(content="Hello John!")
    assert variables == {"greeting": "Hello", "name": "John"}


def test_extract_variables_default_fixture(output_message):
    variables = output_message.extract_variables("Hello John!")
    assert variables == {"greeting": "Hello", "name": "John"}


def test_extract_variables_custom_fixture(output_message):
    def custom_extract_func(
        names: List[str], content_format: str, content: str
    ) -> Dict[str, Any]:
        regex_pattern = re.compile(r"^(?P<greeting>\w+) (?P<name>\w+)!$")
        match = regex_pattern.match(content)
        return match.groupdict() if match else None

    output_message.custom_extract_variables_func = custom_extract_func
    variables = output_message.extract_variables("Hello John!")
    assert variables == {"greeting": "Hello", "name": "John"}


def test_no_role_provided_fixture():
    msg = OutputMessage("{greeting} {name}!")
    assert msg.role == Role.ASSISTANT


def test_extraction_error_fixture(output_message):
    with pytest.raises(ExtractionError):
        output_message.extract_variables("Hello")


class TestInputJSONMessage:
    @pytest.fixture
    def input_json_message(self):
        return InputJSONMessage("{greeting} {user.name}!")

    def test_init(self, input_json_message):
        assert input_json_message._content_format == "{greeting} {user_name}!"
        assert input_json_message._role == Role.USER
        assert input_json_message._const == False

    def test_call(self, input_json_message):
        content = input_json_message(greeting="Hello", user={"name": "John"})
        assert content == "Hello John!"

    def test_missing_variables(self):
        msg = InputJSONMessage(
            "{greeting} {user.name}!", expected_input_varnames={"greeting", "user"}
        )
        with pytest.raises(ExtractionError):
            content = msg(greeting="Hello")

    def test_no_variables(self):
        msg = InputJSONMessage("Hello!", expected_input_varnames=None)
        content = msg()
        assert content == "Hello!"


class TestOutputJSONMessage:
    @pytest.fixture
    def output_json_message(self):
        return OutputJSONMessage("Hello {user.name}! Your age is {user.age}.")

    def test_init(self, output_json_message):
        assert (
            output_json_message._content_format
            == "Hello {user_name}! Your age is {user_age}."
        )
        assert output_json_message._role == Role.ASSISTANT
        assert output_json_message._const == False

    def test_call(self, output_json_message):
        content = "Hello John! Your age is 30."
        extracted_variables = output_json_message(content=content)
        assert extracted_variables == {"user_name": "John", "user_age": "30"}

    def test_extraction_error(self, output_json_message):
        with pytest.raises(ExtractionError):
            output_json_message.extract_variables("Hello")


def test_flatten_dict():
    input_dict = {
        "a": 1,
        "b": {
            "c": 2,
            "d": {
                "e": 3,
            },
        },
    }
    expected_output = {"a": 1, "b.c": 2, "b.d.e": 3}
    assert flatten_dict(input_dict) == expected_output


class TestOutputOptions:
    @pytest.fixture
    def output_messages(self):
        return [
            OutputMessage("Hello {name}! Your age is {age}."),
            OutputMessage("Hi {name}, you are {age} years old."),
        ]

    @pytest.fixture
    def output_options(self, output_messages):
        return OutputOptions(output_messages, role=Role.ASSISTANT)

    def test_init(self, output_options, output_messages):
        assert output_options._output_messages == output_messages
        assert output_options._role == Role.ASSISTANT
        assert output_options._const == False

    def test_extract_variables_success(self, output_options):
        content1 = "Hello John! Your age is 30."
        extracted_variables1 = output_options.extract_variables(content1)
        assert extracted_variables1 == {"name": "John", "age": "30"}

        content2 = "Hi John, you are 30 years old."
        extracted_variables2 = output_options.extract_variables(content2)
        assert extracted_variables2 == {"name": "John", "age": "30"}

    def test_extraction_error(self, output_options):
        content = "Hey John, you're 30."
        with pytest.raises(ExtractionError):
            output_options.extract_variables(content)

    def test_inconsistent_role(self, output_messages):
        output_messages.append(OutputMessage("Hey {name}, age {age}.", role=Role.USER))
        with pytest.raises(ValueError):
            OutputOptions(output_messages, role=Role.ASSISTANT)
