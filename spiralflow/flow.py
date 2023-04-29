from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import copy
from abc import ABC, abstractmethod
import concurrent.futures

from .message import (
    Message,
    InputMessage,
    OutputMessage,
    InputJSONMessage,
    OutputJSONMessage,
    Role,
    ExtractionError,
)
from .chat_history import ChatHistory, ChatHistoryManager
from .chat_llm import ChatLLM
from .memory import Memory


def combine_chat_histories(chat_histories):
    """
    Combines a list of chat histories into one chat history.

    :param chat_histories: List of chat histories to combine.
    :return: Combined chat history.
    """
    return ChatHistory(
        [
            message
            for chat_history in chat_histories
            for message in chat_history.messages
        ]
    )


class BaseFlow(ABC):
    """
    A class abstract class for all flows with inputs and one output at the end.
    """

    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def input_varnames(self) -> Set[str]:
        pass

    @property
    @abstractmethod
    def output_varnames(self) -> Set[str]:
        pass

    @abstractmethod
    def flow(
        self,
    ) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]:
        pass

    @abstractmethod
    def __call__(self) -> Tuple[Dict[str, str], ChatHistory]:
        pass

    @abstractmethod
    def compress_histories(
        self, histories: Tuple[List[ChatHistory], List[ChatHistory]]
    ) -> Tuple[ChatHistory, ChatHistory]:
        pass


class ChatFlow(BaseFlow):
    """
    A class for a chat flow with inputs and outputs at any point (except the first and last message).

    Limitations:
     - Variable checks are done on flow call, not on initialization.
    """

    def __init__(
        self,
        messages: List[Message],
        default_chat_llm: Optional[ChatLLM] = None,
        default_input_chat_history: Optional[ChatHistory] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the ChatFlow class with the given parameters.

        :param messages: List of messages in the chat flow.
        :param default_chat_llm: Optional default chat llm used in flow, if not provided in flow call.
        :param default_input_chat_history: Optional default input chat history used in flow, if not provided in flow call.
        :param verbose: Whether to print verbose output.
        """
        self._messages = [[]]
        self.default_chat_llm = default_chat_llm
        self.default_input_chat_history = default_input_chat_history
        self.verbose = verbose
        self._input_varnames = [set()]
        self._output_varnames = [None]

        for message in messages:
            if isinstance(message, InputMessage):
                self._messages[-1].append(message)
                self._input_varnames[-1].update(message.varnames)
                if self._output_varnames[-1] is not None:
                    self._output_varnames.append(None)
            elif isinstance(message, OutputMessage):
                self._messages[-1].append(message)
                self._output_varnames[-1] = message.varnames
                self._messages.append([])
                self._input_varnames.append(set())
            else:
                raise ValueError(
                    f"Message is not an InputMessage or OutputMessage. All messages in a ChatFlow must be of type InputMessage or OutputMessage. Type: {type(message)}"
                )

        if self._output_varnames[-1] is None:
            raise ValueError(
                f"Last message in a ChatFlow must be of type OutputMessage. Type: {type(self._messages[-1][-1])}"
            )
        self._messages.pop()
        self._input_varnames.pop()

        # might need to be improved
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
    def verbose(self):
        """
        :return: Whether the flow is verbose.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        """
        Sets the verbose attribute.

        :param verbose: Whether the flow is verbose.
        """
        self._verbose = verbose

    @property
    def input_varnames(self):
        """
        :return: A deepcopy of input variable names.
        """
        return set(self._input_varnames[0])

    @property
    def output_varnames(self):
        """
        :return: A deepcopy of output variable names.
        """
        return set(output for outputs in self._output_varnames for output in outputs)

    def flow(
        self,
        input_variables: dict,
        chat_llm: Optional[ChatLLM] = None,
        input_chat_history: Optional[ChatHistory] = None,
    ) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]:
        """
        Runs the chat flow through an LLM.

        :param input_variables: Dictionary of input variables.
        :param chat_llm: Optional chat language model to use for the chat flow.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and a tuple of input and internal chat histories.
        """
        if chat_llm is None:
            chat_llm = self.default_chat_llm
        if chat_llm is None:
            raise ValueError(
                "Chat LLM is missing. Please provide one since there is no default."
            )

        if not self._input_varnames[0].issubset(input_variables.keys()):
            raise ValueError(
                f"Input variables do not match input variable names.\nExpected: {self._input_varnames[0]}\nActual: {input_variables.keys()}"
            )

        if input_chat_history is None:
            input_chat_history = self.default_input_chat_history
        full_chat_history = ChatHistory(
            [] if input_chat_history is None else input_chat_history.messages
        )
        new_input_chat_history = ChatHistory(
            [] if input_chat_history is None else input_chat_history.messages
        )
        internal_chat_histories = []
        all_variables = dict(input_variables)

        for i in range(len(self._messages)):
            messages = []

            for message in full_chat_history.messages:
                messages.append({"role": message.role, "content": message.content})

            internal_chat_history = ChatHistory()
            for message in self._messages[i][:-1]:
                messages.append(
                    {"role": message.role, "content": message(**all_variables)}
                )
                internal_chat_history.add_message(message)
                full_chat_history.add_message(message)

            content, role, _ = chat_llm(messages)

            if role != self._messages[i][-1].role:
                raise ValueError(
                    f"Chat LLM role does not match last message role. {role} != {self._messages[i][-1].role}"
                )

            messages.append({"role": self._messages[i][-1].role, "content": content})

            variables = self._messages[i][-1](content=content)
            all_variables.update(variables)

            full_chat_history.add_message(self._messages[i][-1])

            internal_chat_history.add_message(self._messages[i][-1])
            internal_chat_histories.append(internal_chat_history)

            if self.verbose:
                print(f"Chat flow Step {i}:")
                for message in messages:
                    print(" ", message)

        return all_variables, ([new_input_chat_history], internal_chat_histories)

    def __call__(
        self,
        input_variables: dict,
        chat_llm: Optional[ChatLLM] = None,
        input_chat_history: Optional[ChatHistory] = None,
        return_all: bool = True,
    ) -> Tuple[Dict[str, str], ChatHistory]:
        """
        Runs the chat flow through an LLM.

        :param input_variables: Dictionary of input variables.
        :param chat_llm: Optional chat language model to use for the chat flow.
        :param input_chat_history: Optional input chat history.
        :param return_all: If True, return all variables. If False, return only output variables.
        :return: Tuple of dictionary of output variables and chat history.
        """
        return_variables, histories = self.flow(
            input_variables, chat_llm=chat_llm, input_chat_history=input_chat_history
        )

        histories = self.compress_histories(histories)
        history = ChatHistory(histories[0].messages + histories[1].messages)

        if not return_all:
            return_variables = {
                varname: varvalue
                for varname, varvalue in return_variables.items()
                if varname in self.output_varnames
            }
        return return_variables, history

    def compress_histories(
        self, histories: Tuple[List[ChatHistory], List[ChatHistory]]
    ) -> Tuple[ChatHistory, ChatHistory]:
        """
        Combines a tuple of list of chat histories into a tuple of two chat histories.

        :param histories: Tuple of list of chat histories.
        :return: Tuple of combined input and internal chat histories.
        """
        input_histories, internal_histories = histories

        if len(input_histories) == 0 and len(internal_histories) == 0:
            return ChatHistory(), ChatHistory()

        return (
            input_histories[0],
            ChatHistory(
                [
                    message
                    for chat_history in internal_histories
                    for message in chat_history.messages
                ]
            ),
        )

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
            if "type" in message:
                if message["type"] == "input":
                    mtype = InputMessage
                elif message["type"] == "output":
                    mtype = OutputMessage
                elif message["type"] == "input_json":
                    mtype = InputJSONMessage
                elif message["type"] == "output_json":
                    mtype = OutputJSONMessage
                elif isinstance(message["type"], Message):
                    mtype = message["type"]
                else:
                    raise ValueError(
                        f'Message type must be "input" or "output". Type: {message["type"]}'
                    )
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
                raise ValueError(
                    f'Message role must be Role.USER or Role.ASSISTANT. Role: {message["role"]}'
                )
            content_format = message[role]
            del message[role]
            if "type" in message:
                del message["type"]

            msgs.append(mtype(content_format, role=role, **message))

        return ChatFlow(msgs, **kwargs)


class FuncChatFlow(ChatFlow):
    """
    A class for creating chat flows from functions.
    """

    def __init__(
        self,
        func: Callable[
            [dict, Optional[ChatLLM], Optional[ChatHistory]],
            Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]],
        ],
        input_varnames: Set[str],
        output_varnames: Set[str],
        default_chat_llm: Optional[ChatLLM] = None,
        default_input_chat_history: Optional[ChatHistory] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes a FuncChatFlow.

        :param func: Function to use for the chat flow.
        :param input_varnames: List of input variable names.
        :param output_varnames: List of output variable names.
        :param default_chat_llm: Optional default chat language model used in flow, if not provided in.
        :param default_input_chat_history: Optional default input chat history used in flow, if not provided in.
        :param verbose: If True, print chat flow steps.
        """
        self._func = func
        self._input_varnames = input_varnames
        self._output_varnames = output_varnames
        self.default_chat_llm = default_chat_llm
        self.default_input_chat_history = default_input_chat_history
        self.verbose = verbose

    @property
    def input_varnames(self):
        """
        :return: A deepcopy of input variable names.
        """
        return set(self._input_varnames)

    @property
    def output_varnames(self):
        """
        :return: A deepcopy of output variable names.
        """
        return set(self._output_varnames)

    def flow(
        self,
        input_variables: dict,
        chat_llm: Optional[ChatLLM] = None,
        input_chat_history: Optional[ChatHistory] = None,
    ) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]:
        """
        Runs the chat flow through an LLM.

        :param input_variables: Dictionary of input variables.
        :param chat_llm: Optional chat language model to use for the chat flow.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and a tuple of input and internal chat histories.
        """
        if chat_llm is None:
            chat_llm = self.default_chat_llm
        if chat_llm is None:
            # raise ValueError('Chat LLM is missing. Please provide one since there is no default.')
            # assuming that the chat flow is not using the LLM
            pass

        if not self._input_varnames.issubset(input_variables.keys()):
            raise ValueError(
                f"Input variables do not match input variable names.\nExpected: {self._input_varnames}\nActual: {input_variables.keys()}"
            )

        if input_chat_history is None:
            input_chat_history = self.default_input_chat_history

        outputs, (new_input_chat_histories, internal_histories) = self._func(
            input_variables, chat_llm=chat_llm, input_chat_history=input_chat_history
        )

        if not self._output_varnames.issubset(outputs.keys()):
            raise ValueError(
                f"Output variables do not match output variable names.\nExpected: {self._output_varnames}\nActual: {outputs.keys()}"
            )

        if len(new_input_chat_histories) > 1:
            raise ValueError(
                f"FuncChatFlow func must return a empty or single input chat history. Received {len(new_input_chat_histories)}"
            )

        return outputs, (new_input_chat_histories, internal_histories)


class ChatFlowWrapper(ChatFlow):
    """
    A ChatFlow wrapper class for others to inherit from.
    """

    def __init__(self, chat_flow: ChatFlow, verbose: bool = False) -> None:
        """
        Initializes a ChatFlowWrapper.

        :param chat_flow: ChatFlow to wrap.
        :param verbose: Whether to print verbose output.
        """
        self._chat_flow = chat_flow
        self.verbose = verbose

    @property
    def verbose(self):
        """
        :return: Whether the flow is verbose.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        """
        Sets the verbose attribute.

        :param verbose: Whether the flow is verbose.
        """
        self._chat_flow.verbose = verbose
        self._verbose = verbose

    @property
    def input_varnames(self):
        """
        :return: A deepcopy of input variable names.
        """
        return self._chat_flow.input_varnames

    @property
    def output_varnames(self):
        """
        :return: A deepcopy of output variable names.
        """
        return self._chat_flow.output_varnames

    def flow(
        self,
        input_variables: dict,
        chat_llm: Optional[ChatLLM] = None,
        input_chat_history: Optional[ChatHistory] = None,
    ) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]:
        """
        Runs the chat flow through an LLM.

        :param input_variables: Dictionary of input variables.
        :param chat_llm: Optional chat language model to use for the chat flow.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and a tuple of empty input and internal chat histories.
        """
        variables, histories = self._chat_flow.flow(
            input_variables, chat_llm=chat_llm, input_chat_history=input_chat_history
        )
        return variables, histories


class NoHistory(ChatFlowWrapper):
    """
    A ChatFlow that blocks the input chat history from being passed to the LLM and returns empty input and internal chat histories.
    """

    def __init__(
        self,
        chat_flow: ChatFlow,
        allow_input_history: bool = False,
        allow_rtn_internal_history: bool = False,
        allow_rtn_input_history: bool = False,
        disallow_default_history: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initializes a NoHistory object.

        :param chat_flow: ChatFlow to wrap.
        :param allow_input_history: Whether to allow the input chat history to be passed to the LLM.
        :param allow_rtn_internal_history: Whether to allow the internal chat history to be returned.
        :param allow_rtn_input_history: Whether to allow the input chat history to be returned.
        :param disallow_default_history: Whether to disallow the default chat history to be returned.
        :param verbose: Whether to print verbose output.
        """
        super().__init__(chat_flow, verbose)
        self._allow_input_history = allow_input_history
        self._allow_rtn_internal_history = allow_rtn_internal_history
        self._allow_rtn_input_history = allow_rtn_input_history
        self._disallow_default_history = disallow_default_history

    def flow(
        self,
        input_variables: dict,
        chat_llm: Optional[ChatLLM] = None,
        input_chat_history: Optional[ChatHistory] = None,
    ) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]:
        """
        Runs the chat flow through an LLM.

        :param input_variables: Dictionary of input variables.
        :param chat_llm: Optional chat language model to use for the chat flow.
        :param input_chat_history: Optional input chat history. Will not be used, but internal chat flow may use default.
        :return: Tuple of dictionary of output variables and a tuple of empty input and internal chat histories.
        """
        original_input_chat_history = input_chat_history
        if not self._allow_input_history:
            input_chat_history = None
        if self._disallow_default_history:
            input_chat_history = ChatHistory()

        variables, histories = self._chat_flow.flow(
            input_variables, chat_llm=chat_llm, input_chat_history=input_chat_history
        )

        if not self._allow_rtn_internal_history:
            histories = (histories[0], [])
        if not self._allow_input_history:
            histories = ([original_input_chat_history], histories[1])
        if not self._allow_rtn_input_history:
            histories = ([], histories[1])

        return variables, histories


class NoOldSystemHistory(ChatFlowWrapper):
    def __init__(
        self, chat_flow: ChatFlow, keep_first: bool = False, verbose: bool = False
    ) -> None:
        super().__init__(chat_flow, verbose)
        self.keep_first = keep_first

    def flow(
        self,
        input_variables: dict,
        chat_llm: Optional[ChatLLM] = None,
        input_chat_history: Optional[ChatHistory] = None,
    ) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]:
        if input_chat_history is None:
            mod_input_chat_history = None
        else:
            mod_input_chat_history = ChatHistory()
            sys_count = 0
            for message in input_chat_history.messages:
                if message.role == Role.SYSTEM:
                    if self.keep_first and sys_count == 0:
                        mod_input_chat_history.add_message(message)
                    sys_count += 1
                else:
                    mod_input_chat_history.add_message(message)

        variables, histories = self._chat_flow.flow(
            input_variables,
            chat_llm=chat_llm,
            input_chat_history=mod_input_chat_history,
        )

        return variables, histories


class History(ChatFlowWrapper):
    """
    A class that wraps a ChatFlow and uses a history manager to import and export histories to other
    History Chat Flows.

    Limitations:
     - If importing histories, the input chat histories will be ignored.
    """

    def __init__(
        self,
        chat_flow: ChatFlow,
        history_manager: ChatHistoryManager,
        histories_id: Optional[str],
        histories_ids: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes a History object.

        :param chat_flow: ChatFlow to wrap.
        :param history_manager: Chat history manager to use.
        :param histories_id: Optional ID of the history to use. If provided, this chat flows
                             input and internal histories will be saved to the history manager.
        :param histories_ids: Optional list of IDs of histories to use combine and use.
                              If provided, input chat histories will be ignored.
        """
        super().__init__(chat_flow, verbose)
        self._history_manager = history_manager
        self._histories_id = histories_id
        self._histories_ids = histories_ids

        # Add placeholder
        if self._histories_id is not None:
            self._history_manager.add_chat_history(histories_id + "_input")
            self._history_manager.add_chat_history(histories_id + "_internal")

        # Check placeholders exists for requested histories
        if self._histories_ids is not None:
            specified_histories_ids = []
            for histories_id in self._histories_ids:
                if histories_id in self._history_manager:
                    specified_histories_ids.append(histories_id)
                elif histories_id + "_input" in self._history_manager:
                    specified_histories_ids.append(histories_id + "_input")
                    specified_histories_ids.append(histories_id + "_internal")
                else:
                    raise ValueError("History with id '{histories_id}' does not exist")
            self._histories_ids = specified_histories_ids

    def flow(
        self,
        input_variables: dict,
        chat_llm: Optional[ChatLLM] = None,
        input_chat_history: Optional[ChatHistory] = None,
    ) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]:
        """
        Runs the chat flow through an LLM.

        :param input_variables: Dictionary of input variables.
        :param chat_llm: Optional chat language model to use for the chat flow.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and a tuple of empty input and internal chat histories.
        """
        if self._histories_ids is not None:
            input_chat_history = self._history_manager.get_combined_chat_histories(
                self._histories_ids
            )

        variables, histories = self._chat_flow.flow(
            input_variables, chat_llm=chat_llm, input_chat_history=input_chat_history
        )

        if self._histories_id is not None:
            combined_histories = self.compress_histories(histories)
            self._history_manager.replace_chat_history(
                self._histories_id + "_input", combined_histories[0]
            )
            self._history_manager.replace_chat_history(
                self._histories_id + "_internal", combined_histories[1]
            )

        return variables, histories


class MemoryChatFlow(ChatFlowWrapper):
    """
    A class for creating chat flows that interact with external memories
    """

    def __init__(
        self,
        chat_flow: ChatFlow,
        memory: Memory,
        memory_query_kwargs: Optional[dict] = None,
        default_chat_llm: Optional[ChatLLM] = None,
        default_input_chat_history: Optional[ChatHistory] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes a MemoryChatFlow from a ChatFlow.

        :param chat_flow: ChatFlow to used for the chat flow and to get the query
        :param memory: Memory to use for the chat flow.
        :param memory_query_kwargs: Optional keyword arguments to pass to memory query.
        :param default_chat_llm: Optional default chat language model used in flow, if not provided in.
        :param default_input_chat_history: Optional default input chat history used in flow, if not provided in.
        :param verbose: If True, print chat flow steps.
        """
        super().__init__(chat_flow, verbose)
        self._memory = memory
        self._memory_query_kwargs = memory_query_kwargs
        self.default_chat_llm = default_chat_llm
        self.default_input_chat_history = default_input_chat_history
        # should not mutate varnames, but if chat flow does, this will break, since will not update
        self._input_varnames = self._chat_flow.input_varnames
        self._output_varnames = self._chat_flow.output_varnames
        if "query" not in self._output_varnames:
            raise ValueError(
                'MemoryChatFlow must have an output variable named "query" since it is used to query the memory.'
            )
        if "memory" in self._output_varnames:
            raise ValueError(
                'MemoryChatFlow cannot have an output variable named "memory" since it is reserved for the memory obtained from external memories.'
            )
        self._output_varnames.add("memory")

    @property
    def input_varnames(self):
        """
        :return: A deepcopy of input variable names.
        """
        return set(self._input_varnames)

    @property
    def output_varnames(self):
        """
        :return: A deepcopy of output variable names.
        """
        return set(self._output_varnames)

    def flow(
        self,
        input_variables: dict,
        chat_llm: Optional[ChatLLM] = None,
        input_chat_history: Optional[ChatHistory] = None,
    ) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]:
        """
        Runs the chat flow through an LLM and gets a query which is used to get memory from external memories.

        :param input_variables: Dictionary of input variables.
        :param chat_llm: Optional chat language model to use for the chat flow.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and a tuple of input and internal chat histories.
        """
        if chat_llm is None:
            chat_llm = self.default_chat_llm
        if input_chat_history is None:
            input_chat_history = self.default_input_chat_history

        variables, histories = self._chat_flow.flow(
            input_variables, chat_llm=chat_llm, input_chat_history=input_chat_history
        )

        query = variables["query"]
        memory = self._memory.query(query, **self._memory_query_kwargs)

        variables["memory"] = memory

        return variables, histories


class ConditonalChatFlow(ChatFlowWrapper):
    """
    A class for creating conditional chat flows, which shift flows based on the output of previous messages.
    """

    def __init__(
        self,
        decision_chat_flow: ChatFlow,
        branch_chat_flows: Dict[str, ChatFlow],
        default_chat_llm: Optional[ChatLLM] = None,
        default_input_chat_history: Optional[ChatHistory] = None,
        verbose: bool = False,
    ):
        """
        Initializes a ConditonalChatFlow.

        :param decision_chat_flow: Chat flow for making the decision.
        :param branch_chat_flows: Dictionary of chat flows for each branch. Use `default` as the key for the default branch.
        :param default_chat_llm: Optional default chat language model used in flow, if not provided in flow call.
        :param default_input_chat_history: Optional default input chat history used in flow, if not provided in flow call.
        :param verbose: If True, print chat flow messages.
        """
        self._decision_chat_flow = decision_chat_flow
        self._branch_chat_flows = branch_chat_flows
        self.default_chat_llm = default_chat_llm
        self.default_input_chat_history = default_input_chat_history
        # should not mutate varnames, but if chat flow does, this will break, since will not update
        self._input_varnames = self._decision_chat_flow.input_varnames
        self._output_varnames = self._branch_chat_flows[
            list(self._branch_chat_flows.keys())[0]
        ].output_varnames
        self._output_varnames.update(self._decision_chat_flow.output_varnames)
        self.verbose = verbose

        # check decision branch has a decision variable
        if "decision" not in self._decision_chat_flow.output_varnames:
            raise ValueError(
                'Decision chat flow must have an output variable named "decision".'
            )

        # inputs can be different among the branches update input varnames
        # check that all branches result in the same output variables
        varnames = None
        for branch_chat_flow in self._branch_chat_flows.values():
            self._input_varnames.update(branch_chat_flow.input_varnames)
            if varnames is None:
                varnames = branch_chat_flow.output_varnames
            elif varnames != branch_chat_flow.output_varnames:
                raise ValueError(
                    f"All branch chat flows must have the same output variables.\nExpected: {varnames}\nActual: {branch_chat_flow.output_varnames}"
                )

        # check that outputs of branch are different than outputs of decision
        if len(self._decision_chat_flow.output_varnames.intersection(varnames)) > 0:
            raise ValueError(
                f"Decision chat flow and branch chat flows cannot have the same output variables.\nDecision: {self._decision_chat_flow.output_varnames}\nBranch: {varnames}"
            )

        # check inputs do not contain outputs of decision
        if (
            len(
                self._decision_chat_flow.output_varnames.intersection(
                    self._decision_chat_flow.input_varnames
                )
            )
            > 0
        ):
            raise ValueError(
                f"Decision chat flow cannot have output variables as input variables.\nDecision Input: {self._decision_chat_flow.input_varnames}\nDecision Output: {self._decision_chat_flow.output_varnames}"
            )

    @property
    def verbose(self):
        """
        :return: Whether the flow is verbose.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        """
        Sets the verbose attribute.

        :param verbose: Whether the flow is verbose.
        """
        self._decision_chat_flow.verbose = verbose
        for branch_chat_flow in self._branch_chat_flows.values():
            branch_chat_flow.verbose = verbose
        self._verbose = verbose

    @property
    def input_varnames(self) -> Set[str]:
        return set(self._input_varnames)

    @property
    def output_varnames(self) -> Set[str]:
        return set(self._output_varnames)

    def flow(
        self,
        input_variables: dict,
        chat_llm: Optional[ChatLLM] = None,
        input_chat_history: Optional[ChatHistory] = None,
    ) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]:
        """
        Runs the decision chat flow through an LLM and then from the decision the appropriate branch.

        :param input_variables: Dictionary of input variables.
        :param chat_llm: Optional chat language model to use for the chat flow.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and a tuple of input and internal chat histories.
        """
        if chat_llm is None:
            chat_llm = self.default_chat_llm

        if input_chat_history is None:
            input_chat_history = self.default_input_chat_history
        decision_variables, decision_histories = self._decision_chat_flow.flow(
            input_variables, chat_llm=chat_llm, input_chat_history=input_chat_history
        )

        decision_histories = self._decision_chat_flow.compress_histories(
            decision_histories
        )

        decision = decision_variables["decision"]
        if decision in self._branch_chat_flows:
            branch_chat_flow = self._branch_chat_flows[decision]
        elif "default" in self._branch_chat_flows:
            branch_chat_flow = self._branch_chat_flows["default"]
        else:
            raise ExtractionError(f"Invalid decision: {decision}")

        input_variables = copy.deepcopy(input_variables)
        input_variables.update(decision_variables)

        input_chat_history = ChatHistory(
            decision_histories[0].messages + decision_histories[1].messages
        )
        branch_variables, branch_histories = branch_chat_flow.flow(
            input_variables, chat_llm=chat_llm, input_chat_history=input_chat_history
        )

        branch_histories = branch_chat_flow.compress_histories(branch_histories)

        decision_variables.update(branch_variables)

        # ADD IF RETURN DECISON AND BRANCH OR JUST BRANCH
        return branch_variables, (
            [decision_histories[0], branch_histories[0]],
            [decision_histories[1], branch_histories[1]],
        )

    def compress_histories(
        self, histories: Tuple[List[ChatHistory], List[ChatHistory]]
    ) -> Tuple[ChatHistory, ChatHistory]:
        """
        Combines a tuple of list of chat histories into a tuple of two chat histories.

        :param histories: Tuple of list of chat histories.
        :return: Tuple of combined input and internal chat histories.
        """
        input_histories, internal_histories = histories
        return (
            input_histories[1],  # get the branch input_history
            internal_histories[1],  # get the branch internal_history
        )


class SequentialChatFlows(ChatFlowWrapper):
    """
    A sequential chat flow class that runs a list of chat flows sequentially.

    Limitations:
     - All chat flows use the input history returned by the first chat flow plus internal of previous chat flows.
     - A chat flow can take an input and overwrite the original input with a new output with the same name. Be careful.
    """

    def __init__(
        self,
        chat_flows: List[ChatFlow],
        default_chat_llm: Optional[ChatLLM] = None,
        default_input_chat_history: Optional[ChatHistory] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes a seqeuntial chat flows class.

        :param chat_flows: List of chat flows to run sequentially.
        :param default_chat_llm: Optional default chat language model used in flow, if not provided in flow call.
        :param default_input_chat_history: Optional default input chat history used in flow, if not provided in flow call.
        :param verbose: If True, print chat flow messages.
        """
        self._chat_flows = chat_flows
        self.default_chat_llm = default_chat_llm
        self.default_input_chat_history = default_input_chat_history
        self.verbose = verbose

        # create input_varnames and output_varnames
        # should not mutate varnames, but if a chat flow does, this will break, since will not update
        self._input_varnames = set()
        self._output_varnames = []
        for chat_flow in self._chat_flows:
            self._input_varnames.update(chat_flow.input_varnames)
            self._output_varnames.extend(chat_flow.output_varnames)

        # check no output conflicts
        if len(self._output_varnames) != len(set(self._output_varnames)):
            raise ValueError(
                f"Output variable names conflict between chat flows. Output: {self._output_varnames}"
            )

        self._output_varnames = set(self._output_varnames)

    @property
    def verbose(self):
        """
        :return: Whether the flow is verbose.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        """
        Sets the verbose attribute.

        :param verbose: Whether the flow is verbose.
        """
        for chat_flow in self._chat_flows:
            chat_flow.verbose = verbose
        self._verbose = verbose

    @property
    def input_varnames(self):
        """
        :return: A deepcopy of input variable names.
        """
        return set(self._input_varnames)

    @property
    def output_varnames(self):
        """
        :return: A deepcopy of output variable names.
        """
        return set(self._output_varnames)

    def flow(
        self,
        input_variables: dict,
        chat_llm: Optional[ChatLLM] = None,
        input_chat_history: Optional[ChatHistory] = None,
    ) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]:
        """
        Runs the chat flows through an LLM.

        :param input_variables: Dictionary of input variables.
        :param chat_llm: Optional chat language model to use for the chat flow.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and list of chat histories.
        """
        if chat_llm is None:
            chat_llm = self.default_chat_llm
        if input_chat_history is None:
            input_chat_history = self.default_input_chat_history
        output_variables = {}

        full_chat_history = input_chat_history

        first_input_history = None
        internal_histories = []

        for chat_flow in self._chat_flows:
            chat_flow_output_variables, chat_flow_histories = chat_flow.flow(
                input_variables, chat_llm=chat_llm, input_chat_history=full_chat_history
            )
            input_variables.update(chat_flow_output_variables)
            output_variables.update(chat_flow_output_variables)

            chat_flow_histories = chat_flow.compress_histories(chat_flow_histories)

            if first_input_history is None:
                first_input_history = chat_flow_histories[0]
                full_chat_history = chat_flow_histories[1]
            else:
                full_chat_history = combine_chat_histories(
                    [full_chat_history, chat_flow_histories[1]]
                )

            internal_histories.append(chat_flow_histories[1])

        return output_variables, ([first_input_history], internal_histories)


class ConcurrentChatFlows(ChatFlowWrapper):
    def __init__(
        self,
        chat_flows: List[ChatFlow],
        default_chat_llm: Optional[ChatLLM] = None,
        default_input_chat_history: Optional[ChatHistory] = None,
        max_workers=None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes a concurrent chat flows class.

        :param chat_flows: List of chat flows to run concurrently.
        :param default_chat_llm: Optional default chat language model used in flow, if not provided in flow call.
        :param default_input_chat_history: Optional default input chat history used in flow, if not provided in flow call.
        :param max_workers: Number of threads to use for concurrent chat flows. If None, use all available threads.
        :param verbose: If True, print chat flow messages.
        """
        self._chat_flows = chat_flows
        self.default_chat_llm = default_chat_llm
        self.default_input_chat_history = default_input_chat_history
        self.max_workers = max_workers
        self.verbose = verbose

        # create input_varnames and output_varnames
        # should not mutate varnames, but if a chat flow does, this will break, since will not update
        self._input_varnames = set()
        self._output_varnames = []
        for chat_flow in self._chat_flows:
            self._input_varnames.update(chat_flow.input_varnames)
            self._output_varnames.extend(chat_flow.output_varnames)

        # check no output conflicts and that no outputs are in the inputs
        if len(self._output_varnames) != len(set(self._output_varnames)):
            raise ValueError(
                f"Output variable names conflict between chat flows. Output: {self._output_varnames}"
            )

        self._output_varnames = set(self._output_varnames)

    @property
    def verbose(self):
        """
        :return: Whether the flow is verbose.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        """
        Sets the verbose attribute.

        :param verbose: Whether the flow is verbose.
        """
        for chat_flow in self._chat_flows:
            chat_flow.verbose = verbose
        self._verbose = verbose

    @property
    def input_varnames(self):
        """
        :return: A deepcopy of input variable names.
        """
        return set(self._input_varnames)

    @property
    def output_varnames(self):
        """
        :return: A deepcopy of output variable names.
        """
        return set(self._output_varnames)

    def flow(
        self,
        input_variables: dict,
        chat_llm: Optional[ChatLLM] = None,
        input_chat_history: Optional[ChatHistory] = None,
    ) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]:
        """
        Runs the chat flows concurrently through an LLM.

        :param input_variables: Dictionary of input variables.
        :param chat_llm: Optional chat language model to use for the chat flow.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and tuple of list of chat histories (order matches ordering of chat_flows).
        """
        if chat_llm is None:
            chat_llm = self.default_chat_llm
        if input_chat_history is None:
            input_chat_history = self.default_input_chat_history
        output_variables = {}
        input_histories = []
        internal_histories = []

        def wrapper(chat_flow):
            nonlocal input_variables, chat_llm, input_chat_history
            variables, histories = chat_flow.flow(
                input_variables,
                chat_llm=chat_llm,
                input_chat_history=input_chat_history,
            )
            return variables, chat_flow.compress_histories(histories)

        # Run chat flows concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            for result in executor.map(wrapper, self._chat_flows):
                chat_flow_output_variables, chat_flow_histories = result
                output_variables.update(chat_flow_output_variables)
                input_histories.append(chat_flow_histories[0])
                internal_histories.append(chat_flow_histories[1])

        return output_variables, (input_histories, internal_histories)

    def compress_histories(
        self, histories: Tuple[List[ChatHistory], List[ChatHistory]]
    ) -> Tuple[ChatHistory, ChatHistory]:
        """
        Combines a tuple of list of chat histories into a tuple of two chat histories.

        :param histories: Tuple of list of chat histories.
        :return: Tuple of combined input and internal chat histories.
        """
        input_histories, internal_histories = histories
        if len(input_histories) != len(internal_histories):
            raise ValueError(
                f"Input and internal histories must be the same length. Input: {len(input_histories)}, Internal: {len(internal_histories)}"
            )

        messages = []
        for input_history, internal_history in zip(input_histories, internal_histories):
            messages.extend(input_history.messages)
            messages.extend(internal_history.messages)

        return (ChatHistory(), ChatHistory(messages))


class ChatSpiral(ChatFlowWrapper):
    class Exit(Exception):
        pass

    def __init__(
        self,
        chat_flow: ChatFlow,
        output_varnames_remap: Optional[Dict[str, str]] = None,
        default_chat_llm: Optional[ChatLLM] = None,
        default_input_chat_history: Optional[ChatHistory] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes a chat spiral class.

        :param chat_flow: Chat flow to spiral.
        :param output_varnames_remap: Optional dictionary of output variable names to remap.
        :param default_chat_llm: Optional default chat language model used in flow, if not provided in flow/spiral call.
        :param default_input_chat_history: Optional default input chat history used in flow, if not provided in flow/spiral call.
        """
        super().__init__(chat_flow, verbose=verbose)
        self._output_varnames_remap = output_varnames_remap
        self.default_chat_llm = default_chat_llm
        self.default_input_chat_history = default_input_chat_history

        # check output contains all input variables? No?

    @property
    def output_varnames(self) -> Set[str]:
        output_varnames = {
            self._output_varnames_remap.get(varname, varname)
            for varname in self._chat_flow.output_varnames
        }
        return output_varnames

    def flow(
        self,
        input_variables: dict,
        chat_llm: Optional[ChatLLM] = None,
        input_chat_history: Optional[ChatHistory] = None,
    ) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]:
        """
        Runs the chat flow through an LLM.

        :param input_variables: Dictionary of input variables.
        :param chat_llm: Optional chat language model to use for the chat flow.
        :param input_chat_history: Optional input chat history.
        :return: Tuple of dictionary of output variables and two tuple of list of chat histories.
        """
        if chat_llm is None:
            chat_llm = self.default_chat_llm
        if input_chat_history is None:
            input_chat_history = self.default_input_chat_history

        output_variables, histories = self._chat_flow.flow(
            input_variables, chat_llm=chat_llm, input_chat_history=input_chat_history
        )
        histories = self._chat_flow.compress_histories(histories)
        if self._output_varnames_remap is not None:
            output_variables = {
                self._output_varnames_remap.get(k, k): v
                for k, v in output_variables.items()
            }
        return output_variables, ([histories[0]], [histories[1]])

    def spiral(
        self,
        input_variables: dict,
        reset_history: bool = False,
        chat_llm: Optional[ChatLLM] = None,
        input_chat_history: Optional[ChatHistory] = None,
        max_iterations: Optional[int] = None,
    ) -> Tuple[Dict[str, str], ChatHistory]:
        """
        Runs the chat flow through an LLM continuously.

        :param input_variables: Dictionary of input variables.
        :param reset_history: Whether to reset the chat history after each chat flow completion.
        :param chat_llm: Optional chat language model to use for the chat flow.
        :param input_chat_history: Optional input chat history.
        :param max_iterations: Maximum number of iterations to run through the chat flow.

        :return: Tuple of dictionary of output variables and chat history
        """
        variables = dict(input_variables)
        chat_history = input_chat_history
        try:
            count = 0
            while True:
                if max_iterations is not None and count >= max_iterations:
                    raise ChatSpiral.Exit

                new_variables, histories = self.flow(
                    variables, chat_llm=chat_llm, input_chat_history=chat_history
                )
                variables.update(new_variables)
                if not reset_history:
                    chat_history = ChatHistory(
                        histories[0][0].messages + histories[1][0].messages
                    )

                count += 1
        except ChatSpiral.Exit:
            return variables, chat_history

    def __call__(
        self,
        input_variables: dict,
        reset_history: bool = False,
        chat_llm: Optional[ChatLLM] = None,
        input_chat_history: Optional[ChatHistory] = None,
        max_iterations: Optional[int] = None,
        return_all: bool = True,
    ) -> Tuple[Dict[str, str], ChatHistory]:
        """
        Runs the chat flow through an LLM continuously.

        :param input_variables: Dictionary of input variables.
        :param reset_history: Whether to reset the chat history after each chat flow completion.
        :param chat_llm: Optional chat language model to use for the chat flow.
        :param input_chat_history: Optional input chat history.
        :param max_iterations: Maximum number of iterations to run through the chat flow.
        :param return_all: Whether to return all output variables.

        :return: Tuple of dictionary of output variables and chat history
        """
        variables, history = self.spiral(
            input_variables,
            reset_history=reset_history,
            chat_llm=chat_llm,
            input_chat_history=input_chat_history,
            max_iterations=max_iterations,
        )

        if not return_all:
            variables = {
                k: v for k, v in variables.items() if k in self.output_varnames
            }
        return variables, history

    def compress_histories(
        self, histories: Tuple[List[ChatHistory], List[ChatHistory]]
    ) -> Tuple[ChatHistory, ChatHistory]:
        """
        Combines a tuple of list of chat histories into a tuple of two chat histories.

        :param histories: Tuple of list of chat histories.
        :return: Tuple of combined input and internal chat histories.
        """
        input_histories, internal_histories = histories
        return (
            input_histories[0][0],
            internal_histories[0][0],
        )
