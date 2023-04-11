from typing import List
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
        self.messages = []
        if messages is not None:
            for message in messages:
                self.add_message(message)

    def add_message(self, message: Message) -> None:
        """
        Adds a message made constant to the chat history.

        :param message: The message to add to the chat history.
        """
        if not message.defined():
            raise ValueError('Message must have all defined variables.')
        self.messages.append(message.get_const())
