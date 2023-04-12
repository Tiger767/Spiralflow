import copy
from typing import Any, List

from message import Message


class ChatHistory:
    """
    A class to store the chat history (constant messages) for a chat flow/session.
    """

    def __init__(self, messages: List[Message] = None) -> None:
        """
        Initializes the ChatHistory class with the given parameters.

        :param messages: The messages to initialize the chat history with.
        """
        self._messages = []
        self._const = False
        if messages is not None:
            for message in messages:
                self.add_message(message)

    @property
    def messages(self) -> List[Message]:
        """
        Gets the messages in the chat history.

        :return: The messages in the chat history.
        """
        return list(self._messages)

    def add_message(self, message: Message) -> None:
        """
        Adds a message made constant to the chat history.

        :param message: The message to add to the chat history.
        """
        if not message.defined():
            raise ValueError("Message must have all defined variables.")
        if self._const:
            raise ValueError("Cannot add messages to a constant chat history.")
        self._messages.append(message.get_const())

    def make_const(self) -> None:
        """
        Makes this chat history constant so messages cannot be added.
        """
        self._const = True

    def get_const(self) -> Any:
        """
        Creates a deepcopy of self and makes it constant.

        :return: A deepcopy of this chat history made constant so messages cannot be added
        """
        chat_history = copy.deepcopy(self)
        chat_history.make_const()
        return chat_history
