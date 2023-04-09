from typing import Dict, List, Optional, Set, Tuple
import copy

from message import Message, InputMessage, OutputMessage
from chat_history import ChatHistory
from chat_llm import ChatLLM


class SimpleChatFlow:
    def __init__(self, messages: List[Message]) -> None:
        """
        :param messages: List of messages in the chat flow.
        """
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
    def input_varnames(self) -> Set[str]:
        return copy.deepcopy(self._input_varnames)
    
    @property
    def output_varnames(self) -> Set[str]:
        return copy.deepcopy(self._output_varnames)

    def flow(self, chat_llm: ChatLLM, input_variables: dict, input_chat_history: Optional[ChatHistory] = None) -> Tuple[Dict[str, str], List[ChatHistory]]:
        """
        :param chat_llm: The chat language model to use for the chat flow.
        :param input_variables: Dictionary of input variables.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and a chat history in a list.
        """
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

        content, role = chat_llm(messages)
        if role != self.messages[-1].role:
            raise ValueError('Chat LLM role does not match last message role.')
        messages.append({'role': self.messages[-1].role, 'content': content})
        variables = self.messages[-1](content=content)
        internal_chat_history.add_message(self.messages[-1])

        return variables, [internal_chat_history]

    def __call__(self, chat_llm: ChatLLM, input_variables: dict, input_chat_history: Optional[ChatHistory] = None) -> Tuple[Dict[str, str], ChatHistory]:
        """
        :param chat_llm: The chat language model to use for the chat flow.
        :param input_variables: Dictionary of input variables.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and chat history.
        """
        variables, internal_chat_histories = self.flow(chat_llm, input_variables, input_chat_history)
        return variables, internal_chat_histories[0]


class ChatFlow(SimpleChatFlow):
    def __init__(self, messages: List[Message]) -> None:
        """
        :param messages: List of messages in the chat flow.
        """
        self.messages = [[]]
        self._input_varnames = [set()]
        self._output_varnames = [None]

        for message in messages:
            if isinstance(message, InputMessage):
                self.messages[-1].append(message)
                self._input_varnames[-1].update(message.varnames)
                if self._output_varnames[-1] is not None:
                    self._output_varnames.append(None)
            elif isinstance(message, OutputMessage):
                self.messages[-1].append(message)
                self._output_varnames[-1] = message.varnames
                self.messages.append([])
                self._input_varnames.append(set())
            else:
                raise ValueError(f'Message is not an InputMessage or OutputMessage. All messages in a ChatFlow must be of type InputMessage or OutputMessage.')

        if self._output_varnames[-1] is None:
            raise ValueError(f'Last message in a ChatFlow must be of type OutputMessage.')
        self.messages.pop()
        self._input_varnames.pop()

        if len(self._input_varnames) > 1:
            for i in range(1, len(self._input_varnames)):
                for varname in self._input_varnames[i]:
                    # Check if the varname exists in any previous output_varnames
                    for j in range(i):
                        if varname in self._output_varnames[j]:
                            break
                    else:
                        # Add the varname to the first input_varnames
                        self._input_varnames[0].add(varname)

    @property
    def input_varnames(self):
        return copy.deepcopy(self._input_varnames[0])
    
    @property
    def output_varnames(self):
        return copy.deepcopy(self._output_varnames[-1])

    def flow(self, chat_llm: ChatLLM, input_variables: dict, input_chat_history: Optional[ChatHistory] = None) -> Tuple[Dict[str, str], List[ChatHistory]]:
        """
        :param chat_llm: The chat language model to use for the chat flow.
        :param input_variables: Dictionary of input variables.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and chat history.
        """
        if not self._input_varnames[0].issubset(input_variables.keys()):
            raise ValueError('Input variables do not match input variable names.')

        full_chat_history = ChatHistory([] if input_chat_history is None else input_chat_history.messages)
        internal_chat_histories = []
        all_variables = {}
        all_variables.update(input_variables)

        for i in range(len(self.messages)):
            messages = []

            for message in full_chat_history.messages:
                messages.append({'role': message.role, 'content': message.content})

            internal_chat_history = ChatHistory()
            for message in self.messages[i][:-1]:
                messages.append({'role': message.role, 'content': message(**all_variables)})
                internal_chat_history.add_message(message)

            content, role = chat_llm(messages)
            if role != self.messages[i][-1].role:
                raise ValueError('Chat LLM role does not match last message role.')
            messages.append({'role': self.messages[i][-1].role, 'content': content})
            variables = self.messages[i][-1](content=content)
            all_variables.update(variables)
            internal_chat_history.add_message(self.messages[i][-1])

            full_chat_history.add_message(self.messages[i][-1])
            internal_chat_histories.append(internal_chat_history)
        return variables, internal_chat_histories
        
    def __call__(self, chat_llm: ChatLLM, input_variables: dict, input_chat_history: Optional[ChatHistory] = None, return_all: bool = True) -> Tuple[Dict[str, str], ChatHistory]:
        """
        :param chat_llm: The chat language model to use for the chat flow.
        :param input_variables: Dictionary of input variables.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and chat history.
        """
        all_variables, internal_chat_histories = self.flow(chat_llm, input_variables, input_chat_history)
        if return_all:
            return_variables = all_variables
        else: 
            return_variables = {varname: varvalue for varname, varvalue in all_variables.items() if varname in self.output_varnames}
        internal_chat_history = ChatHistory([message for internal_chat_history in internal_chat_histories for message in internal_chat_history.messages])
        return return_variables, internal_chat_history
