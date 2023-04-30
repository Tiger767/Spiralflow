import pytest
from unittest.mock import patch
from spiralflow.chat_llm import ChatLLM
from spiralflow.message import Message


# Helper function to create a sample message
def create_sample_message():
    return Message(user_id="user1", content="Hello", timestamp="2023-04-29T12:34:56")


# TestChatLLM
class TestChatLLM:
    def test_init(self):
        # Test with default parameters
        chat_llm = ChatLLM()
        assert chat_llm.gpt_model == "gpt-3.5-turbo"
        assert chat_llm.stream == False
        assert chat_llm.model_params == {}

        # Test with custom parameters
        custom_params = {"temperature": 0.8, "max_tokens": 50}
        chat_llm = ChatLLM(gpt_model="custom_model", stream=True, **custom_params)
        assert chat_llm.gpt_model == "custom_model"
        assert chat_llm.stream == True
        assert chat_llm.model_params == custom_params

    @patch("openai.ChatCompletion.create")
    def test_call(self, chat_completion_mock):
        messages = [create_sample_message()]

        # Mock OpenAI API response
        chat_completion_mock.return_value = {
            "choices": [{"message": {"content": "Hi there!", "role": "assistant"}}]
        }

        chat_llm = ChatLLM()

        # Test with a list of messages
        content, role, response = chat_llm(messages)
        assert content == "Hi there!"
        assert role == "assistant"
        assert response == chat_completion_mock.return_value

        # Test with stream mode (expect NotImplementedError)
        chat_llm.stream = True
        with pytest.raises(NotImplementedError):
            chat_llm(messages)
