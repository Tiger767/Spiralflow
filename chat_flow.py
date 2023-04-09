from typing import List, Optional
import copy

from message import Message, InputMessage, OutputMessage
from chat_history import ChatHistory
from chat_llm import ChatLLM


class ChatFlow:
    def __init__(self, messages: List[Message]):
        self.messages = messages
        self._input_varnames = set()

        for message in self.messages[:-1]:
            if not isinstance(message, InputMessage):
                raise ValueError(f'Message is not an InputMessage. All messages besides the last in a ChatFlow must be of type InputMessage.')
            self._input_varnames.update(message.varnames)

        if not isinstance(self.messages[-1], OutputMessage):
            raise ValueError(f'Message is not an OutPutMessage. Last message in a ChatFlow must be of type OutputMessage.')
        self._output_varnames = set(self.messages[-1].varnames)

    @property
    def input_varnames(self):
        return copy.deepcopy(self._input_varnames)
    
    @property
    def output_varnames(self):
        return copy.deepcopy(self._output_varnames)

    def flow(self, chat_llm: ChatLLM, input_variables: dict, input_chat_history: Optional[ChatHistory] = None):
        if not self._input_varnames.issubset(input_variables.keys()):
            raise ValueError('Input variables do not match input variable names.')

        messages = []
        if input_chat_history is not None:
            for message in self.input_chat_history.messages:
                messages.append({'role': message.role, 'content': message.content})

        internal_chat_history = ChatHistory()
        for message in self.messages[:-1]:
            messages.append({'role': message.role, 'content': message(**input_variables)})
            internal_chat_history.add_message(message)

        content = chat_llm(messages)
        messages.append({'role': self.messages[-1].role, 'content': content})
        variables = self.messages[-1](content=content)
        internal_chat_history.add_message(self.messages[-1])

        return variables, internal_chat_history

    def __call__(self, chat_llm: ChatLLM, input_variables: dict, input_chat_history: Optional[ChatHistory] = None):
        return self.flow(chat_llm, input_variables, input_chat_history)
