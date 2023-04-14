import copy
from typing import Any, Dict, List, Optional

from .message import Message


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


class ChatHistoryManager:
    """
    A class to manage chat histories for multiple chat flows.
    """

    def __init__(self) -> None:
        """
        Initializes the ChatHistoryManager class.
        """
        self._chat_histories = {}

    def get_chat_history(self, chat_id: str) -> ChatHistory:
        """
        :param chat_id: The chat ID to get the chat history for.
        :return: The chat history for the given chat ID.
        """
        if chat_id not in self._chat_histories:
            raise KeyError(f"Chat history for chat ID {chat_id} does not exist.")
            # self._chat_histories[chat_id] = ChatHistory()
        if self._chat_histories[chat_id] is None:
            raise ValueError(f"Chat history for chat ID {chat_id} is None.")
        return self._chat_histories[chat_id]

    def add_chat_history(
        self, chat_id: str, chat_history: Optional[ChatHistory] = None
    ) -> None:
        """
        :param chat_id: The chat ID to add the chat history for.
        :param chat_history: The chat history to add for the given chat ID.
                             If not provided, a placeholder (None) is added.
        """
        if chat_id in self._chat_histories:
            raise KeyError(f"Chat history for chat ID {chat_id} already exists.")
        self._chat_histories[chat_id] = chat_history

    def replace_chat_history(self, chat_id: str, chat_history: ChatHistory) -> None:
        """
        :param chat_id: The chat ID to replace the chat history for.
        :param chat_history: The chat history to replace for the given chat ID.
        """
        if chat_id not in self._chat_histories:
            raise KeyError(f"Chat history for chat ID {chat_id} does not exist.")
        self._chat_histories[chat_id] = chat_history

    def delete_chat_history(self, chat_id: str) -> None:
        """
        :param chat_id: The chat ID to delete the chat history for.
        """
        if chat_id not in self._chat_histories:
            raise KeyError(f"Chat history for chat ID {chat_id} does not exist.")
        del self._chat_histories[chat_id]

    def get_chat_histories(self) -> Dict[str, ChatHistory]:
        """
        :return: The chat histories for all chat IDs.
        """
        return dict(self._chat_histories)

    def clear_chat_histories(self) -> None:
        """
        Clears all chat histories.
        """
        self._chat_histories = {}

    def get_combined_chat_histories(self, chat_ids: List[str]) -> ChatHistory:
        """
        :param chat_ids: The chat IDs to get the combined chat history for. (Order matters)
        :return: The combined chat history for the given chat IDs.
        """
        chat_messages = []
        for chat_id in chat_ids:
            chat_history = self.get_chat_history(chat_id)
            chat_messages.extend(chat_history.messages)

        return ChatHistory(chat_messages)

    def __len__(self) -> int:
        return len(self._chat_histories)

    def __contains__(self, chat_id: str) -> bool:
        return chat_id in self._chat_histories

    def __iter__(self):
        return iter(self._chat_histories)

    def __getitem__(self, chat_id: str) -> ChatHistory:
        return self.get_chat_history(chat_id)

    def __setitem__(self, chat_id: str, chat_history: ChatHistory) -> None:
        self.add_chat_history(chat_id, chat_history)

    def __delitem__(self, chat_id: str) -> None:
        self.delete_chat_history(chat_id)
