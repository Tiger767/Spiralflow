import pytest
from spiralflow.flow import ChatFlow, InputMessage, OutputMessage, Role
from spiralflow.chat_llm import ChatLLM
from spiralflow.chat_history import ChatHistory


class TestChatFlow:
    @pytest.fixture
    def chat_llm(self):
        return ChatLLM()

    @pytest.fixture
    def input_chat_history(self):
        return ChatHistory()

    @pytest.fixture
    def messages(self):
        return [
            InputMessage("{greeting}", role=Role.USER, varnames=["greeting"]),
            OutputMessage("Hello! How can I help you today?"),
            InputMessage("{question}", role=Role.USER, varnames=["question"]),
            OutputMessage(
                "I can answer your question about {topic}.", varnames=["topic"]
            ),
        ]

    @pytest.fixture
    def chat_flow(self, messages, chat_llm, input_chat_history):
        return ChatFlow(
            messages,
            default_chat_llm=chat_llm,
            default_input_chat_history=input_chat_history,
            verbose=True,
        )

    def test_init(self, chat_flow):
        assert chat_flow is not None

    def test_verbose(self, chat_flow):
        assert chat_flow.verbose is True
        chat_flow.verbose = False
        assert chat_flow.verbose is False

    def test_input_varnames(self, chat_flow):
        assert chat_flow.input_varnames == {"greeting", "question"}

    def test_output_varnames(self, chat_flow):
        assert chat_flow.output_varnames == {"topic"}

    def test_flow(self, chat_flow, chat_llm, input_chat_history):
        input_variables = {"greeting": "Hi", "question": "What is AI?"}
        output_variables, histories = chat_flow.flow(
            input_variables, chat_llm=chat_llm, input_chat_history=input_chat_history
        )
        assert "topic" in output_variables
        assert len(histories[0]) == 1
        assert len(histories[1]) == 1

    def test_call(self, chat_flow, chat_llm, input_chat_history):
        input_variables = {"greeting": "Hi", "question": "What is AI?"}
        output_variables, history = chat_flow(
            input_variables, chat_llm=chat_llm, input_chat_history=input_chat_history
        )
        assert "topic" in output_variables
        assert len(history.messages) == 4

    def test_compress_histories(self, chat_flow, input_chat_history):
        input_histories = [input_chat_history]
        internal_histories = [ChatHistory()]
        compressed_histories = chat_flow.compress_histories(
            (input_histories, internal_histories)
        )
        assert len(compressed_histories[0].messages) == 0
        assert len(compressed_histories[1].messages) == 0

    def test_from_dicts(self, chat_llm, input_chat_history):
        messages = [
            {"type": "input", Role.USER: "{greeting}", "varnames": ["greeting"]},
            {"type": "output", Role.ASSISTANT: "Hello! How can I help you today?"},
            {"type": "input", Role.USER: "{question}", "varnames": ["question"]},
            {
                "type": "output",
                Role.ASSISTANT: "I can answer your question about {topic}.",
                "varnames": ["topic"],
            },
        ]
        chat_flow = ChatFlow.from_dicts(
            messages,
            default_chat_llm=chat_llm,
            default_input_chat_history=input_chat_history,
        )
        assert chat_flow is not None
        assert len(chat_flow._messages) == 2
        assert chat_flow._input_varnames[0] == {"greeting", "question"}
        assert chat_flow._output_varnames[0] == {"topic"}
