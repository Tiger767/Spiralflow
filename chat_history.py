from typing import List
from message import Message


class ChatHistory:
    def __init__(self, messages: List[Message] = None) -> None:
        self.messages = []
        if messages is not None:
            for message in messages:
                self.add_message(message)

    def add_message(self, message: Message) -> None:
        """
        :param message: The message to add to the chat history.
        """
        if not message.defined():
            raise ValueError('Message must have all defined variables.')
        self.messages.append(message.get_const())
