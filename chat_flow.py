from typing import Any, Dict, List, Optional, Set, Tuple
import copy

from message import InputJSONMessage, Message, InputMessage, OutputJSONMessage, OutputMessage, Role
from chat_history import ChatHistory
from chat_llm import ChatLLM


class SimpleChatFlow:
    """
    A class for a simple chat flow with inputs and one output at the end.
    """

    def __init__(self, messages: List[Message], verbose: bool = False) -> None:
        """
        Initializes the SimpleChatFlow class with the given parameters.

        :param messages: List of messages in the chat flow.
        :param verbose: Whether to print verbose output.
        """
        self.messages = messages
        self.verbose = verbose
        self._input_varnames = set()

        for message in self.messages[:-1]:
            if not isinstance(message, InputMessage):
                raise ValueError(f'Message is not an InputMessage. All messages besides the last in a ChatFlow must be of type InputMessage. Type: {type(message)}')
            self._input_varnames.update(message.varnames)

        if not isinstance(self.messages[-1], OutputMessage):
            raise ValueError(f'Message is not an OutPutMessage. Last message in a ChatFlow must be of type OutputMessage. Type: {type(self.messages[-1])}')
        self._output_varnames = set(self.messages[-1].varnames)

    @property
    def input_varnames(self) -> Set[str]:
        return copy.deepcopy(self._input_varnames)
    
    @property
    def output_varnames(self) -> Set[str]:
        return copy.deepcopy(self._output_varnames)

    def flow(self, chat_llm: ChatLLM, input_variables: dict, input_chat_history: Optional[ChatHistory] = None) -> Tuple[Dict[str, str], List[ChatHistory]]:
        """
        Runs the chat flow through an LLM.

        :param chat_llm: The chat language model to use for the chat flow.
        :param input_variables: Dictionary of input variables.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and a chat history in a list.
        """
        if not self._input_varnames.issubset(input_variables.keys()):
            raise ValueError(f'Input variables do not match input variable names.\nExpected: {self._input_varnames}\nActual: {input_variables.keys()}.')

        messages = []
        if input_chat_history is not None:
            for message in self.input_chat_history.messages:
                messages.append({'role': message.role, 'content': message.content})

        internal_chat_history = ChatHistory()
        for message in self.messages[:-1]:
            messages.append({'role': message.role, 'content': message(**input_variables)})
            internal_chat_history.add_message(message)

        content, role, _ = chat_llm(messages)
        if role != self.messages[-1].role:
            raise ValueError(f'Chat LLM role does not match last message role. {role} != {self.messages[-1].role}')
        messages.append({'role': self.messages[-1].role, 'content': content})

        if self.verbose:
            print('Chat flow:')
            for message in messages:
                print(' ', message)

        variables = self.messages[-1](content=content)
        internal_chat_history.add_message(self.messages[-1])

        return variables, [internal_chat_history]

    def __call__(self, chat_llm: ChatLLM, input_variables: dict, input_chat_history: Optional[ChatHistory] = None) -> Tuple[Dict[str, str], ChatHistory]:
        """
        Runs the chat flow through an LLM.

        :param chat_llm: The chat language model to use for the chat flow.
        :param input_variables: Dictionary of input variables.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and chat history.
        """
        variables, internal_chat_histories = self.flow(chat_llm, input_variables, input_chat_history)
        return variables, internal_chat_histories[0]


class ChatFlow(SimpleChatFlow):
    """
    A class for a chat flow with inputs and outputs at any point (except the first and last message).
    """
    
    def __init__(self, messages: List[Message], verbose: bool = False) -> None:
        """
        Initializes the ChatFlow class with the given parameters.

        :param messages: List of messages in the chat flow.
        """
        self.messages = [[]]
        self.verbose = verbose
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
                raise ValueError(f'Message is not an InputMessage or OutputMessage. All messages in a ChatFlow must be of type InputMessage or OutputMessage. Type: {type(message)}')

        if self._output_varnames[-1] is None:
            raise ValueError(f'Last message in a ChatFlow must be of type OutputMessage. Type: {type(self.messages[-1][-1])}')
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
        Runs the chat flow through an LLM.

        :param chat_llm: The chat language model to use for the chat flow.
        :param input_variables: Dictionary of input variables.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and chat history.
        """
        if not self._input_varnames[0].issubset(input_variables.keys()):
            raise ValueError(f'Input variables do not match input variable names.\nExpected: {self._input_varnames[0]}\nActual: {input_variables.keys()}')

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
                full_chat_history.add_message(message)

            content, role, _ = chat_llm(messages)

            if role != self.messages[i][-1].role:
                raise ValueError(f'Chat LLM role does not match last message role. {role} != {self.messages[i][-1].role}')

            messages.append({'role': self.messages[i][-1].role, 'content': content})

            variables = self.messages[i][-1](content=content)
            all_variables.update(variables)

            full_chat_history.add_message(self.messages[i][-1])

            internal_chat_history.add_message(self.messages[i][-1])
            internal_chat_histories.append(internal_chat_history)

            if self.verbose:
                print(f'Chat flow Step {i}:')
                for message in messages:
                    print(' ', message)

        return variables, internal_chat_histories
        
    def __call__(self, chat_llm: ChatLLM, input_variables: dict, input_chat_history: Optional[ChatHistory] = None, return_all: bool = True) -> Tuple[Dict[str, str], ChatHistory]:
        """
        Runs the chat flow through an LLM.

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

    @staticmethod
    def from_dicts(messages: List[Dict], **kwargs) -> None:
        """
        Creates a ChatFlow from a list of dictionaries of messages with metadata.

        :param messages: List of dictionaries of messages {role: content_format, type: input/output} in the chat flow.
        :return: ChatFlow object with the messages.
        """
        msgs = []
        for message in messages:
            mtype = None
            role = None
            if 'type' in message:
                if message['type'] == 'input':
                    mtype = InputMessage
                elif message['type'] == 'output':
                    mtype = OutputMessage
                elif message['type'] == 'input_json':
                    mtype = InputJSONMessage
                elif message['type'] == 'output_json':
                    mtype = OutputJSONMessage
                elif isinstance(message['type'], Message):
                    mtype = message['type']
                else:
                    raise ValueError(f'Message type must be "input" or "output". Type: {message["type"]}')
            if Role.USER in message:
                role = Role.USER
                if mtype is None:
                    mtype = InputMessage
            elif Role.ASSISTANT in message:
                role = Role.ASSISTANT
                if mtype is None:
                    mtype = OutputMessage
            elif Role.SYSTEM in message:
                role = Role.SYSTEM
                mtype = InputMessage
            else:
                raise ValueError(f'Message role must be Role.USER or Role.ASSISTANT. Role: {message["role"]}')
            content_format = message[role]
            del message[role]
            if 'type' in message:
                del message['type']

            msgs.append(mtype(content_format, role=role, **message))

        return ChatFlow(msgs, **kwargs)


class ChatFlowManager:
    """
    A class for managing chat flows.
    """
    
    class ChatFlowNode:
        """
        A class for adding mapping information to a chat flow.
        """
        
        def __init__(self, name, chat_flow: ChatFlow, input_varnames_map: Optional[Dict[str, str]] = None,
                     output_varnames_map: Optional[Dict[str, str]] = None,
                     input_chat_history_map: Optional[List[Dict[str, Any]]] = None):
            """
            Initializes a ChatFlowNode.

            :param name: Name of the chat flow.
            :param chat_flow: Chat flow.
            :param input_varnames_map: Optional dictionary of input variable names to map to the chat flow.
            :param output_varnames_map: Optional dictionary of output variable names to map to the chat flow.
            :param input_chat_history_map: Optional list of dictionaries of input chat history metadata to map to the chat flow.
            """
            self.name = name
            self.chat_flow = chat_flow
            self.input_varnames_map = {} if input_varnames_map is None else input_varnames_map
            self.output_varnames_map = {} if output_varnames_map is None else output_varnames_map
            self.input_chat_history_map = {} if input_chat_history_map is None else input_chat_history_map

            # Need to add validation checks


        def flow(self, chat_llm: ChatLLM, input_variables: dict, input_chat_histories: Dict[str, Dict[str, ChatHistory]]) -> Tuple[Dict[str, str], List[ChatHistory]]:
            """
            Runs the chat flow through an LLM while remapping input variables, chat histories, and output variables.

            :param chat_llm: The chat language model to use for the chat flow.
            :param input_variables: Dictionary of input variables.
            :param input_chat_histories: Dictionary of input chat histories.
            :return: Tuple of dictionary of output variables and chat histories.
            """
            # need to add validation checks

            input_variables = {self.input_varnames_map.get(varname, varname): varvalue for varname, varvalue in input_variables.items()}
            
            messages = []
            for history_metadata in self.input_chat_history_map:
                chat_histories = input_chat_histories[history_metadata['name']]
                chat_history = chat_histories[history_metadata['type']]
                ndxs = history_metadata['ndxs'] if 'ndxs' in history_metadata else range(len(chat_history.messages))
                for ndx in ndxs:
                    messages.append(chat_history.messages[ndx])

            input_chat_history = ChatHistory(messages)

            output_variables, chat_histories = self.chat_flow(chat_llm, input_variables, input_chat_history)

            output_variables = {self.output_varnames_map.get(varname, varname): varvalue for varname, varvalue in output_variables.items()}
            
            return output_variables, chat_histories

    def __init__(self, chat_flow_nodes: List[ChatFlowNode]):
        """
        Initializes a ChatFlowManager.

        :param chat_flow_nodes: List of ChatFlowNodes in sequential order.
        """
        self.chat_flows_nodes = chat_flow_nodes

        # need to add validation checks?

    def flow(self, chat_llms: Dict[str, ChatLLM],
             input_variables: dict,
             input_chat_histories: Dict[str, ChatHistory]) -> Tuple[Dict[str, str], List[ChatHistory]]:
        """
        Runs all the chat flows through the LLMs while remapping input variables, chat histories, and output variables.

        :param chat_llms: Dictionary of chat language models with names mapping to chat flow node names.
        :param input_variables: Dictionary of input variables.
        :param input_chat_histories: Dictionary of input chat histories with names (do not need to be same as chat flow node names)
        :return: Tuple of dictionary of output variables and chat histories.
        """

        # need to add validation checks

        input_chat_histories = {name: {'initial': chat_history} for name, chat_history in input_chat_histories.items()}

        all_variables = {}
        internal_chat_histories = []
        for chat_flow_node in self.chat_flows_nodes:
            variables, internal_chat_history = chat_flow_node.flow(chat_llms[chat_flow_node.name], input_variables, input_chat_histories)
            
            all_variables.update(variables)

            internal_chat_histories.append(internal_chat_history)
            
            if chat_flow_node.name in input_chat_histories:
                input_chat_histories[chat_flow_node.name]['internal'] = internal_chat_history
            else:
                input_chat_histories[chat_flow_node.name] = {'internal': internal_chat_history}

        return all_variables, internal_chat_histories
