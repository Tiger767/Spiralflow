from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import copy
from abc import ABC, abstractmethod
import concurrent.futures

from message import (
    Message,
    InputMessage,
    OutputMessage,
    InputJSONMessage,
    OutputJSONMessage,
    Role,
    ExtractionError,
)
from chat_history import ChatHistory
from chat_llm import ChatLLM


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
        all_variables = {}
        all_variables.update(input_variables)

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

        return variables, ([new_input_chat_history], internal_chat_histories)

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

        :param input_histories: List of input chat histories.
        :param internal_histories: List of internal chat histories.
        :return: Tuple of combined input and internal chat histories.
        """
        input_histories, internal_histories = histories
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
        default_chat_llm: Optional[ChatHistory] = None,
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

        if len(new_input_chat_histories) != 1:
            raise ValueError(
                f"FuncChatFlow func must return a single input chat history. Received {len(new_input_chat_histories)}"
            )

        # if len(new_input_chat_histories) != len(internal_histories):
        #    raise ValueError(f'FuncChatFlow func must return the same number of input and internal chat histories. Received {len(new_input_chat_histories)} and {len(internal_histories)}')

        return outputs, (new_input_chat_histories, internal_histories)


class ConditonalChatFlow(ChatFlow):
    """
    A class for creating conditional chat flows, which shift flows based on the output of previous messages.
    """

    def __init__(
        self,
        decision_chat_flow: ChatFlow,
        branch_chat_flows: Dict[str, ChatFlow],
        share_input_history: bool = True,
        share_internal_history: bool = True,
        default_chat_llm: Optional[ChatLLM] = None,
        default_input_chat_history: Optional[ChatHistory] = None,
        verbose: bool = False,
    ):
        """
        Initializes a ConditonalChatFlow.

        :param decision_chat_flow: Chat flow for making the decision.
        :param branch_chat_flows: Dictionary of chat flows for each branch.
        :param share_input_history: If True, share the input chat history between the decision and branch chat flows.
        :param share_internal_history: If True, share the internal chat history between the decision and branch chat flows.
        :param default_chat_llm: Optional default chat language model used in flow, if not provided in flow call.
        :param default_input_chat_history: Optional default input chat history used in flow, if not provided in flow call.
        :param verbose: If True, print chat flow messages.
        """
        self._decision_chat_flow = decision_chat_flow
        self._branch_chat_flows = branch_chat_flows
        self._share_input_history = share_input_history
        self._share_internal_history = share_internal_history
        self.default_chat_llm = default_chat_llm
        self.default_input_chat_history = default_input_chat_history
        self._input_varnames = self._decision_chat_flow.input_varnames
        self._output_varnames = self._branch_chat_flows[
            self._branch_chat_flows.keys()[0]
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
        decision_variables, decision_histories = self._decision_chat_flow(
            input_variables, chat_llm=chat_llm, input_chat_history=input_chat_history
        )

        decision_histories = self._decision_chat_flow.compress_histories(
            decision_histories
        )

        decision = decision_variables["decision"]
        try:
            branch_chat_flow = self._branch_chat_flows[decision]
        except KeyError:
            raise ExtractionError(f"Invalid decision: {decision}")

        input_variables = copy.deepcopy(input_variables)
        input_variables.update(decision_variables)

        messages = []
        if self._share_input_history:
            messages += input_chat_history.messages
        if self._share_internal_history:
            messages += decision_histories[1].messages
        input_chat_history = ChatHistory(messages) if len(messages) > 0 else None
        branch_variables, branch_histories = branch_chat_flow(
            input_variables, chat_llm=chat_llm, input_chat_history=input_chat_history
        )

        branch_histories = branch_chat_flow.compress_histories(branch_histories)

        branch_variables.update(decision_variables)

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

        :param input_histories: List of input chat histories.
        :param internal_histories: List of internal chat histories.
        :return: Tuple of combined input and internal chat histories.
        """
        input_histories, internal_histories = histories
        return (
            input_histories[1],  # get the branch input_history
            internal_histories[1],  # get the branch internal_history
        )


class SequentialChatFlows(ChatFlow):
    """
    A sequential chat flow class that runs a list of chat flows sequentially.

    Limitations:
     - All chat flows use the input history returned by the first chat flow plus internal of previous chat flows.
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
        if len(self._input_varnames.intersection(self._output_varnames)) > 0:
            raise ValueError(
                f"Some output variable names are also in the input variable names.\nInput: {self._input_varnames}\nOutput: {self._output_varnames}"
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
            chat_flow_output_variables, chat_flow_histories = chat_flow(
                input_variables, chat_llm=chat_llm, input_chat_history=full_chat_history
            )
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


class ConcurrentChatFlows(ChatFlow):
    def __init__(
        self,
        chat_flows: List[ChatFlow],
        default_chat_llm: Optional[ChatLLM] = None,
        default_input_chat_history: Optional[ChatHistory] = None,
        num_threads=None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes a concurrent chat flows class.

        :param chat_flows: List of chat flows to run concurrently.
        :param default_chat_llm: Optional default chat language model used in flow, if not provided in flow call.
        :param default_input_chat_history: Optional default input chat history used in flow, if not provided in flow call.
        :param num_threads: Number of threads to use for concurrent chat flows. If None, use all available threads.
        :param verbose: If True, print chat flow messages.
        """
        self._chat_flows = chat_flows
        self.default_chat_llm = default_chat_llm
        self.default_input_chat_history = default_input_chat_history
        self.num_threads = num_threads
        self.verbose = verbose

        # create input_varnames and output_varnames
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
            variables, histories = chat_flow(
                input_variables,
                chat_llm=chat_llm,
                input_chat_history=input_chat_history,
            )
            return variables, chat_flow.compress_histories(histories)

        # Run chat flows concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_threads
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

        :param input_histories: List of input chat histories.
        :param internal_histories: List of internal chat histories.
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


class ChatSpiral(ChatFlow):
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
        self._chat_flow = chat_flow
        self._output_varnames_remap = output_varnames_remap
        self._output_varnames = self._chat_flow.output_varnames
        self.default_chat_llm = default_chat_llm
        self.default_input_chat_history = default_input_chat_history
        self.verbose = verbose

        # check output contains all input variables? No?

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
    def input_varnames(self) -> Set[str]:
        return set(self._input_variables.keys())

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

        output_variables, histories = self._chat_flow(
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
    ) -> Tuple[Dict[str, str], ChatHistory]:
        variables = dict(input_variables)
        chat_history = input_chat_history
        try:
            while True:
                new_variables, histories = self.flow(
                    variables, chat_llm=chat_llm, input_chat_history=chat_history
                )
                variables.update(new_variables)
                if not reset_history:
                    chat_history = ChatHistory(
                        histories[0][0].messages + histories[1][0].messages
                    )
        except ChatSpiral.Exit:
            return variables, chat_history

    def __call__(
        self, chat_llm: ChatLLM, return_all: bool = True
    ) -> Tuple[Dict[str, str], ChatHistory]:
        variables, histories = self.spiral(chat_llm)

        histories = self.compress_histories(histories)
        history = ChatHistory(histories[0].messages + histories[1].messages)

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

        :param input_histories: List of input chat histories.
        :param internal_histories: List of internal chat histories.
        :return: Tuple of combined input and internal chat histories.
        """
        input_histories, internal_histories = histories
        return (
            input_histories[0][0],
            internal_histories[0][0],
        )


class ChatFlowManager:
    """
    A class for managing chat flows.
    """

    class ChatFlowNode:
        """
        A class for adding mapping information to a chat flow.
        """

        def __init__(
            self,
            name,
            chat_flow: ChatFlow,
            input_varnames_remap: Optional[Dict[str, str]] = None,
            output_varnames_remap: Optional[Dict[str, str]] = None,
            input_chat_history_remap: Optional[List[Dict[str, Any]]] = None,
        ):
            """
            Initializes a ChatFlowNode.

            :param name: Name of the chat flow.
            :param chat_flow: Chat flow.
            :param input_varnames_remap: Optional dictionary of input variable names to remap to the chat flow.
            :param output_varnames_remap: Optional dictionary of output variable names to remap to the chat flow.
            :param input_chat_history_remap: Optional list of dictionaries of input chat history metadata to remap to the chat flow.
            """
            self.name = name
            self.chat_flow = chat_flow
            self.input_varnames_remap = (
                {} if input_varnames_remap is None else input_varnames_remap
            )
            self.output_varnames_remap = (
                {} if output_varnames_remap is None else output_varnames_remap
            )
            self.input_chat_history_remap = (
                {} if input_chat_history_remap is None else input_chat_history_remap
            )

            for value in self.input_varnames_remap.values():
                if value not in self.chat_flow.input_varnames:
                    raise ValueError(
                        f"Invalid input map value. {value} not in chat flow input. Expected one of {self.chat_flow.input_varnames}"
                    )

            for key in self.output_varnames_remap:
                if key not in self.chat_flow.output_varnames:
                    raise ValueError(
                        f"Invalid output map key. {key} not in chat flow output. Expected one of {self.chat_flow.output_varnames}"
                    )

        def flow(
            self,
            input_variables: dict,
            chat_llm: Optional[ChatLLM] = None,
            input_chat_histories: Dict[str, Dict[str, ChatHistory]] = None,
        ) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]:
            """
            Runs the chat flow through an LLM while remapping input variables, chat histories, and output variables.

            :param input_variables: Dictionary of input variables.
            :param chat_llm: Optional chat language model to use for the chat flow.
            :param input_chat_histories: Optional dictionary of input chat histories.
            :return: Tuple of dictionary of output variables and chat histories.
            """
            input_variables = {
                self.input_varnames_remap.get(varname, varname): varvalue
                for varname, varvalue in input_variables.items()
            }

            if input_chat_histories is None:
                input_chat_history = None
            else:
                messages = []
                for history_metadata in self.input_chat_history_remap:
                    chat_histories = input_chat_histories[history_metadata["name"]]
                    chat_history = chat_histories[history_metadata["type"]]
                    ndxs = (
                        history_metadata["ndxs"]
                        if "ndxs" in history_metadata
                        else range(len(chat_history.messages))
                    )
                    for ndx in ndxs:
                        messages.append(chat_history.messages[ndx])

                input_chat_history = ChatHistory(messages)

            output_variables, chat_histories = self.chat_flow(
                input_variables,
                chat_llm=chat_llm,
                input_chat_history=input_chat_history,
            )

            output_variables = {
                self.output_varnames_remap.get(varname, varname): varvalue
                for varname, varvalue in output_variables.items()
            }

            return output_variables, chat_histories

    def __init__(self, chat_flow_nodes: List[ChatFlowNode]):
        """
        Initializes a ChatFlowManager.

        :param chat_flow_nodes: List of ChatFlowNodes in sequential order.
        """
        self.chat_flows_nodes = chat_flow_nodes

        self._output_varnames = []
        all_input_varnames = set()

        for node in self.chat_flows_nodes:
            node_input_varnames = set(
                node.input_varnames_remap.keys()
            )  # maps do not have to contain all values so not full check
            node_output_varnames = set(node.output_varnames_remap.values())  # ditto

            self._output_varnames.extend(node_output_varnames)
            all_input_varnames.update(node_input_varnames)

        if len(self._output_varnames) != len(set(self._output_varnames)):
            raise ValueError("Output variable names conflict between chat flow nodes.")

        self._output_varnames = set(self._output_varnames)

        if len(all_input_varnames.intersection(node_output_varnames)) > 0:
            raise ValueError(
                "Input variable names conflict with output variable names."
            )

    def flow(
        self,
        input_variables: dict,
        chat_llms: Optional[Dict[str, ChatLLM]] = None,
        input_chat_histories: Optional[Dict[str, ChatHistory]] = None,
    ) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]:
        """
        Runs all the chat flows through the LLMs while remapping input variables, chat histories, and output variables.

        :param input_variables: Dictionary of input variables.
        :param chat_llms: Dictionary of chat language models with names mapping to chat flow node names.
        :param input_chat_histories: Dictionary of input chat histories with names (do not need to be same as chat flow node names)
        :return: Tuple of dictionary of output variables and chat histories.
        """

        # need to add validation checks

        input_chat_histories = {
            name: {"initial": chat_history}
            for name, chat_history in input_chat_histories.items()
        }

        all_variables = {}
        internal_chat_histories = []
        for chat_flow_node in self.chat_flows_nodes:
            variables, internal_chat_history = chat_flow_node.flow(
                input_variables,
                chat_llm=chat_llms.get(chat_flow_node.name, None),
                input_chat_histories=input_chat_histories,
            )

            all_variables.update(variables)

            internal_chat_histories.append(internal_chat_history)

            if chat_flow_node.name in input_chat_histories:
                input_chat_histories[chat_flow_node.name][
                    "internal"
                ] = internal_chat_history
            else:
                input_chat_histories[chat_flow_node.name] = {
                    "internal": internal_chat_history
                }

        return all_variables, internal_chat_histories
