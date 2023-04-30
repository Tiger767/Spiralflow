import pytest
from spiralflow.chat_history import ChatHistory, ChatHistoryManager, Message


# Helper function to create a sample message
def create_sample_message():
    return Message(user_id="user1", content="Hello", timestamp="2023-04-29T12:34:56")


# TestChatHistory
class TestChatHistory:
    def test_init(self):
        chat_history = ChatHistory()
        assert len(chat_history.messages) == 0

        messages = [create_sample_message()]
        chat_history = ChatHistory(messages=messages)
        assert len(chat_history.messages) == 1

    def test_messages_property(self):
        messages = [create_sample_message()]
        chat_history = ChatHistory(messages=messages)
        assert chat_history.messages == messages

    def test_add_message(self):
        message = create_sample_message()
        chat_history = ChatHistory()

        # Test with a message that has undefined variables
        message._content = None
        with pytest.raises(ValueError):
            chat_history.add_message(message)

        # Test with a constant chat history
        message._content = "Hello"
        chat_history.make_const()
        with pytest.raises(ValueError):
            chat_history.add_message(message)

        # Test with a valid message
        chat_history = ChatHistory()
        chat_history.add_message(message)
        assert len(chat_history.messages) == 1

    def test_make_const(self):
        chat_history = ChatHistory()
        chat_history.make_const()
        assert chat_history._const == True

    def test_get_const(self):
        chat_history = ChatHistory()
        const_chat_history = chat_history.get_const()
        assert const_chat_history._const == True
        assert const_chat_history._messages == chat_history._messages


# TestChatHistoryManager
class TestChatHistoryManager:
    def test_init(self):
        manager = ChatHistoryManager()
        assert len(manager) == 0

    def test_get_chat_history(self):
        manager = ChatHistoryManager()

        # Test with non-existent chat_id
        with pytest.raises(KeyError):
            manager.get_chat_history("non_existent_chat_id")

        # Test with valid chat_id
        manager.add_chat_history("chat1")
        chat_history = manager.get_chat_history("chat1")
        assert chat_history is not None

    def test_add_chat_history(self):
        manager = ChatHistoryManager()

        # Test with existing chat_id
        manager.add_chat_history("chat1")
        with pytest.raises(KeyError):
            manager.add_chat_history("chat1")

        # Test with valid chat_id
        manager.add_chat_history("chat2")
        assert "chat2" in manager

    def test_replace_chat_history(self):
        manager = ChatHistoryManager()
        manager.add_chat_history("chat1")

        # Test with non-existent chat_id
        with pytest.raises(KeyError):
            manager.replace_chat_history("non_existent_chat_id", ChatHistory())

        # Test with valid chat_id
        new_chat_history = ChatHistory(messages=[create_sample_message()])
        manager.replace_chat_history("chat1", new_chat_history)
        assert manager.get_chat_history("chat1") == new_chat_history

    def test_delete_chat_history(self):
        manager = ChatHistoryManager()
        manager.add_chat_history("chat1")

        # Test with non-existent chat_id
        with pytest.raises(KeyError):
            manager.delete_chat_history("non_existent_chat_id")

        # Test with valid chat_id
        manager.delete_chat_history("chat1")
        assert "chat1" not in manager

    def test_get_chat_histories(self):
        manager = ChatHistoryManager()
        manager.add_chat_history("chat1")
        manager.add_chat_history("chat2")

        chat_histories = manager.get_chat_histories()
        assert "chat1" in chat_histories
        assert "chat2" in chat_histories

    def test_clear_chat_histories(self):
        manager = ChatHistoryManager()
        manager.add_chat_history("chat1")
        manager.add_chat_history("chat2")

        manager.clear_chat_histories()
        assert len(manager) == 0

    def test_get_combined_chat_histories(self):
        manager = ChatHistoryManager()
        chat_history1 = ChatHistory(messages=[create_sample_message()])
        chat_history2 = ChatHistory(messages=[create_sample_message()])
        manager.add_chat_history("chat1", chat_history1)
        manager.add_chat_history("chat2", chat_history2)

        combined_chat_history = manager.get_combined_chat_histories(["chat1", "chat2"])
        assert len(combined_chat_history.messages) == 2

    def test_magic_methods(self):
        manager = ChatHistoryManager()

        # Test __len__
        assert len(manager) == 0
        manager.add_chat_history("chat1")
        assert len(manager) == 1

        # Test __contains__
        assert "chat1" in manager

        # Test __iter__
        manager.add_chat_history("chat2")
        chat_ids = set()
        for chat_id in manager:
            chat_ids.add(chat_id)
        assert chat_ids == {"chat1", "chat2"}

        # Test __getitem__
        chat_history = manager["chat1"]
        assert chat_history is not None

        # Test __setitem__
        new_chat_history = ChatHistory(messages=[create_sample_message()])
        manager["chat3"] = new_chat_history
        assert "chat3" in manager
        assert manager["chat3"] == new_chat_history
