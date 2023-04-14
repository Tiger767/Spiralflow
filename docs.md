<a id="chat_history"></a>

# chat\_history

<a id="chat_history.ChatHistory"></a>

## ChatHistory Objects

```python
class ChatHistory()
```

A class to store the chat history (constant messages) for a chat flow/session.

<a id="chat_history.ChatHistory.__init__"></a>

#### \_\_init\_\_

```python
def __init__(messages: List[Message] = None) -> None
```

Initializes the ChatHistory class with the given parameters.

**Arguments**:

- `messages`: The messages to initialize the chat history with.

<a id="chat_history.ChatHistory.messages"></a>

#### messages

```python
@property
def messages() -> List[Message]
```

Gets the messages in the chat history.

**Returns**:

The messages in the chat history.

<a id="chat_history.ChatHistory.add_message"></a>

#### add\_message

```python
def add_message(message: Message) -> None
```

Adds a message made constant to the chat history.

**Arguments**:

- `message`: The message to add to the chat history.

<a id="chat_history.ChatHistory.make_const"></a>

#### make\_const

```python
def make_const() -> None
```

Makes this chat history constant so messages cannot be added.

<a id="chat_history.ChatHistory.get_const"></a>

#### get\_const

```python
def get_const() -> Any
```

Creates a deepcopy of self and makes it constant.

**Returns**:

A deepcopy of this chat history made constant so messages cannot be added

<a id="chat_history.ChatHistoryManager"></a>

## ChatHistoryManager Objects

```python
class ChatHistoryManager()
```

A class to manage chat histories for multiple chat flows.

<a id="chat_history.ChatHistoryManager.__init__"></a>

#### \_\_init\_\_

```python
def __init__() -> None
```

Initializes the ChatHistoryManager class.

<a id="chat_history.ChatHistoryManager.get_chat_history"></a>

#### get\_chat\_history

```python
def get_chat_history(chat_id: str) -> ChatHistory
```

**Arguments**:

- `chat_id`: The chat ID to get the chat history for.

**Returns**:

The chat history for the given chat ID.

<a id="chat_history.ChatHistoryManager.add_chat_history"></a>

#### add\_chat\_history

```python
def add_chat_history(chat_id: str,
                     chat_history: Optional[ChatHistory] = None) -> None
```

**Arguments**:

- `chat_id`: The chat ID to add the chat history for.
- `chat_history`: The chat history to add for the given chat ID.
If not provided, a placeholder (None) is added.

<a id="chat_history.ChatHistoryManager.replace_chat_history"></a>

#### replace\_chat\_history

```python
def replace_chat_history(chat_id: str, chat_history: ChatHistory) -> None
```

**Arguments**:

- `chat_id`: The chat ID to replace the chat history for.
- `chat_history`: The chat history to replace for the given chat ID.

<a id="chat_history.ChatHistoryManager.delete_chat_history"></a>

#### delete\_chat\_history

```python
def delete_chat_history(chat_id: str) -> None
```

**Arguments**:

- `chat_id`: The chat ID to delete the chat history for.

<a id="chat_history.ChatHistoryManager.get_chat_histories"></a>

#### get\_chat\_histories

```python
def get_chat_histories() -> Dict[str, ChatHistory]
```

**Returns**:

The chat histories for all chat IDs.

<a id="chat_history.ChatHistoryManager.clear_chat_histories"></a>

#### clear\_chat\_histories

```python
def clear_chat_histories() -> None
```

Clears all chat histories.

<a id="chat_history.ChatHistoryManager.get_combined_chat_histories"></a>

#### get\_combined\_chat\_histories

```python
def get_combined_chat_histories(chat_ids: List[str]) -> ChatHistory
```

**Arguments**:

- `chat_ids`: The chat IDs to get the combined chat history for. (Order matters)

**Returns**:

The combined chat history for the given chat IDs.

<a id="chat_llm"></a>

# chat\_llm

<a id="chat_llm.ChatLLM"></a>

## ChatLLM Objects

```python
class ChatLLM()
```

A class for chat completion using the GPT model.

<a id="chat_llm.ChatLLM.__init__"></a>

#### \_\_init\_\_

```python
def __init__(gpt_model: str = "gpt-3.5-turbo", stream=False, **kwargs) -> None
```

Initializes the ChatLLM class with the given parameters.

**Arguments**:

- `gpt_model`: GPT model to use for chat completion.
- `stream`: Whether to use stream mode.

<a id="chat_llm.ChatLLM.__call__"></a>

#### \_\_call\_\_

```python
def __call__(messages: List[Message]) -> Tuple[str, str, Dict]
```

Generates a response using the GPT model based on the input messages.

**Arguments**:

- `messages`: List of messages to use for chat completion.

**Returns**:

Response from the chat completion with content, role, and metadata.

<a id="flow"></a>

# flow

<a id="flow.combine_chat_histories"></a>

#### combine\_chat\_histories

```python
def combine_chat_histories(chat_histories)
```

Combines a list of chat histories into one chat history.

**Arguments**:

- `chat_histories`: List of chat histories to combine.

**Returns**:

Combined chat history.

<a id="flow.BaseFlow"></a>

## BaseFlow Objects

```python
class BaseFlow(ABC)
```

A class abstract class for all flows with inputs and one output at the end.

<a id="flow.ChatFlow"></a>

## ChatFlow Objects

```python
class ChatFlow(BaseFlow)
```

A class for a chat flow with inputs and outputs at any point (except the first and last message).

Limitations:
 - Variable checks are done on flow call, not on initialization.

<a id="flow.ChatFlow.__init__"></a>

#### \_\_init\_\_

```python
def __init__(messages: List[Message],
             default_chat_llm: Optional[ChatLLM] = None,
             default_input_chat_history: Optional[ChatHistory] = None,
             verbose: bool = False) -> None
```

Initializes the ChatFlow class with the given parameters.

**Arguments**:

- `messages`: List of messages in the chat flow.
- `default_chat_llm`: Optional default chat llm used in flow, if not provided in flow call.
- `default_input_chat_history`: Optional default input chat history used in flow, if not provided in flow call.
- `verbose`: Whether to print verbose output.

<a id="flow.ChatFlow.verbose"></a>

#### verbose

```python
@property
def verbose()
```

**Returns**:

Whether the flow is verbose.

<a id="flow.ChatFlow.verbose"></a>

#### verbose

```python
@verbose.setter
def verbose(verbose: bool)
```

Sets the verbose attribute.

**Arguments**:

- `verbose`: Whether the flow is verbose.

<a id="flow.ChatFlow.input_varnames"></a>

#### input\_varnames

```python
@property
def input_varnames()
```

**Returns**:

A deepcopy of input variable names.

<a id="flow.ChatFlow.output_varnames"></a>

#### output\_varnames

```python
@property
def output_varnames()
```

**Returns**:

A deepcopy of output variable names.

<a id="flow.ChatFlow.flow"></a>

#### flow

```python
def flow(
    input_variables: dict,
    chat_llm: Optional[ChatLLM] = None,
    input_chat_history: Optional[ChatHistory] = None
) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]
```

Runs the chat flow through an LLM.

**Arguments**:

- `input_variables`: Dictionary of input variables.
- `chat_llm`: Optional chat language model to use for the chat flow.
- `input_chat_history`: Optional input chat history.

**Returns**:

Tuple of dictionary of output variables and a tuple of input and internal chat histories.

<a id="flow.ChatFlow.__call__"></a>

#### \_\_call\_\_

```python
def __call__(input_variables: dict,
             chat_llm: Optional[ChatLLM] = None,
             input_chat_history: Optional[ChatHistory] = None,
             return_all: bool = True) -> Tuple[Dict[str, str], ChatHistory]
```

Runs the chat flow through an LLM.

**Arguments**:

- `input_variables`: Dictionary of input variables.
- `chat_llm`: Optional chat language model to use for the chat flow.
- `input_chat_history`: Optional input chat history.
- `return_all`: If True, return all variables. If False, return only output variables.

**Returns**:

Tuple of dictionary of output variables and chat history.

<a id="flow.ChatFlow.compress_histories"></a>

#### compress\_histories

```python
def compress_histories(
    histories: Tuple[List[ChatHistory], List[ChatHistory]]
) -> Tuple[ChatHistory, ChatHistory]
```

Combines a tuple of list of chat histories into a tuple of two chat histories.

**Arguments**:

- `histories`: Tuple of list of chat histories.

**Returns**:

Tuple of combined input and internal chat histories.

<a id="flow.ChatFlow.from_dicts"></a>

#### from\_dicts

```python
@staticmethod
def from_dicts(messages: List[Dict], **kwargs) -> None
```

Creates a ChatFlow from a list of dictionaries of messages with metadata.

**Arguments**:

- `messages`: List of dictionaries of messages {role: content_format, type: input/output} in the chat flow.

**Returns**:

ChatFlow object with the messages.

<a id="flow.FuncChatFlow"></a>

## FuncChatFlow Objects

```python
class FuncChatFlow(ChatFlow)
```

A class for creating chat flows from functions.

<a id="flow.FuncChatFlow.__init__"></a>

#### \_\_init\_\_

```python
def __init__(func: Callable[[dict, Optional[ChatLLM], Optional[ChatHistory]],
                            Tuple[Dict[str, str], Tuple[List[ChatHistory],
                                                        List[ChatHistory]]], ],
             input_varnames: Set[str],
             output_varnames: Set[str],
             default_chat_llm: Optional[ChatLLM] = None,
             default_input_chat_history: Optional[ChatHistory] = None,
             verbose: bool = False) -> None
```

Initializes a FuncChatFlow.

**Arguments**:

- `func`: Function to use for the chat flow.
- `input_varnames`: List of input variable names.
- `output_varnames`: List of output variable names.
- `default_chat_llm`: Optional default chat language model used in flow, if not provided in.
- `default_input_chat_history`: Optional default input chat history used in flow, if not provided in.
- `verbose`: If True, print chat flow steps.

<a id="flow.FuncChatFlow.input_varnames"></a>

#### input\_varnames

```python
@property
def input_varnames()
```

**Returns**:

A deepcopy of input variable names.

<a id="flow.FuncChatFlow.output_varnames"></a>

#### output\_varnames

```python
@property
def output_varnames()
```

**Returns**:

A deepcopy of output variable names.

<a id="flow.FuncChatFlow.flow"></a>

#### flow

```python
def flow(
    input_variables: dict,
    chat_llm: Optional[ChatLLM] = None,
    input_chat_history: Optional[ChatHistory] = None
) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]
```

Runs the chat flow through an LLM.

**Arguments**:

- `input_variables`: Dictionary of input variables.
- `chat_llm`: Optional chat language model to use for the chat flow.
- `input_chat_history`: Optional input chat history.

**Returns**:

Tuple of dictionary of output variables and a tuple of input and internal chat histories.

<a id="flow.ChatFlowWrapper"></a>

## ChatFlowWrapper Objects

```python
class ChatFlowWrapper(ChatFlow)
```

A ChatFlow wrapper class for others to inherit from.

<a id="flow.ChatFlowWrapper.__init__"></a>

#### \_\_init\_\_

```python
def __init__(chat_flow: ChatFlow, verbose: bool = False) -> None
```

Initializes a ChatFlowWrapper.

**Arguments**:

- `chat_flow`: ChatFlow to wrap.
- `verbose`: Whether to print verbose output.

<a id="flow.ChatFlowWrapper.verbose"></a>

#### verbose

```python
@property
def verbose()
```

**Returns**:

Whether the flow is verbose.

<a id="flow.ChatFlowWrapper.verbose"></a>

#### verbose

```python
@verbose.setter
def verbose(verbose: bool)
```

Sets the verbose attribute.

**Arguments**:

- `verbose`: Whether the flow is verbose.

<a id="flow.ChatFlowWrapper.input_varnames"></a>

#### input\_varnames

```python
@property
def input_varnames()
```

**Returns**:

A deepcopy of input variable names.

<a id="flow.ChatFlowWrapper.output_varnames"></a>

#### output\_varnames

```python
@property
def output_varnames()
```

**Returns**:

A deepcopy of output variable names.

<a id="flow.ChatFlowWrapper.flow"></a>

#### flow

```python
def flow(
    input_variables: dict,
    chat_llm: Optional[ChatLLM] = None,
    input_chat_history: Optional[ChatHistory] = None
) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]
```

Runs the chat flow through an LLM.

**Arguments**:

- `input_variables`: Dictionary of input variables.
- `chat_llm`: Optional chat language model to use for the chat flow.
- `input_chat_history`: Optional input chat history.

**Returns**:

Tuple of dictionary of output variables and a tuple of empty input and internal chat histories.

<a id="flow.NoHistory"></a>

## NoHistory Objects

```python
class NoHistory(ChatFlowWrapper)
```

A ChatFlow that blocks the input chat history from being passed to the LLM and returns empty input and internal chat histories.

<a id="flow.NoHistory.__init__"></a>

#### \_\_init\_\_

```python
def __init__(chat_flow: ChatFlow,
             allow_input_history: bool = False,
             allow_rtn_internal_history: bool = False,
             allow_rtn_input_history: bool = False,
             disallow_default_history: bool = False,
             verbose: bool = False) -> None
```

Initializes a NoHistory object.

**Arguments**:

- `chat_flow`: ChatFlow to wrap.
- `allow_input_history`: Whether to allow the input chat history to be passed to the LLM.
- `allow_rtn_internal_history`: Whether to allow the internal chat history to be returned.
- `allow_rtn_input_history`: Whether to allow the input chat history to be returned.
- `disallow_default_history`: Whether to disallow the default chat history to be returned.
- `verbose`: Whether to print verbose output.

<a id="flow.NoHistory.flow"></a>

#### flow

```python
def flow(
    input_variables: dict,
    chat_llm: Optional[ChatLLM] = None,
    input_chat_history: Optional[ChatHistory] = None
) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]
```

Runs the chat flow through an LLM.

**Arguments**:

- `input_variables`: Dictionary of input variables.
- `chat_llm`: Optional chat language model to use for the chat flow.
- `input_chat_history`: Optional input chat history. Will not be used, but internal chat flow may use default.

**Returns**:

Tuple of dictionary of output variables and a tuple of empty input and internal chat histories.

<a id="flow.History"></a>

## History Objects

```python
class History(ChatFlowWrapper)
```

A class that wraps a ChatFlow and uses a history manager to import and export histories to other
History Chat Flows.

Limitations:
 - If importing histories, the input chat histories will be ignored.

<a id="flow.History.__init__"></a>

#### \_\_init\_\_

```python
def __init__(chat_flow: ChatFlow,
             history_manager: ChatHistoryManager,
             histories_id: Optional[str],
             histories_ids: Optional[List[str]] = None,
             verbose: bool = False) -> None
```

Initializes a History object.

**Arguments**:

- `chat_flow`: ChatFlow to wrap.
- `history_manager`: Chat history manager to use.
- `histories_id`: Optional ID of the history to use. If provided, this chat flows
input and internal histories will be saved to the history manager.
- `histories_ids`: Optional list of IDs of histories to use combine and use.
If provided, input chat histories will be ignored.

<a id="flow.History.flow"></a>

#### flow

```python
def flow(
    input_variables: dict,
    chat_llm: Optional[ChatLLM] = None,
    input_chat_history: Optional[ChatHistory] = None
) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]
```

Runs the chat flow through an LLM.

**Arguments**:

- `input_variables`: Dictionary of input variables.
- `chat_llm`: Optional chat language model to use for the chat flow.
- `input_chat_history`: Optional input chat history.

**Returns**:

Tuple of dictionary of output variables and a tuple of empty input and internal chat histories.

<a id="flow.MemoryChatFlow"></a>

## MemoryChatFlow Objects

```python
class MemoryChatFlow(ChatFlowWrapper)
```

A class for creating chat flows that interact with external memories

<a id="flow.MemoryChatFlow.__init__"></a>

#### \_\_init\_\_

```python
def __init__(chat_flow: ChatFlow,
             memory: Memory,
             memory_query_kwargs: Optional[dict] = None,
             default_chat_llm: Optional[ChatLLM] = None,
             default_input_chat_history: Optional[ChatHistory] = None,
             verbose: bool = False) -> None
```

Initializes a MemoryChatFlow from a ChatFlow.

**Arguments**:

- `chat_flow`: ChatFlow to used for the chat flow and to get the query
- `memory`: Memory to use for the chat flow.
- `memory_query_kwargs`: Optional keyword arguments to pass to memory query.
- `default_chat_llm`: Optional default chat language model used in flow, if not provided in.
- `default_input_chat_history`: Optional default input chat history used in flow, if not provided in.
- `verbose`: If True, print chat flow steps.

<a id="flow.MemoryChatFlow.input_varnames"></a>

#### input\_varnames

```python
@property
def input_varnames()
```

**Returns**:

A deepcopy of input variable names.

<a id="flow.MemoryChatFlow.output_varnames"></a>

#### output\_varnames

```python
@property
def output_varnames()
```

**Returns**:

A deepcopy of output variable names.

<a id="flow.MemoryChatFlow.flow"></a>

#### flow

```python
def flow(
    input_variables: dict,
    chat_llm: Optional[ChatLLM] = None,
    input_chat_history: Optional[ChatHistory] = None
) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]
```

Runs the chat flow through an LLM and gets a query which is used to get memory from external memories.

**Arguments**:

- `input_variables`: Dictionary of input variables.
- `chat_llm`: Optional chat language model to use for the chat flow.
- `input_chat_history`: Optional input chat history.

**Returns**:

Tuple of dictionary of output variables and a tuple of input and internal chat histories.

<a id="flow.ConditonalChatFlow"></a>

## ConditonalChatFlow Objects

```python
class ConditonalChatFlow(ChatFlowWrapper)
```

A class for creating conditional chat flows, which shift flows based on the output of previous messages.

<a id="flow.ConditonalChatFlow.__init__"></a>

#### \_\_init\_\_

```python
def __init__(decision_chat_flow: ChatFlow,
             branch_chat_flows: Dict[str, ChatFlow],
             default_chat_llm: Optional[ChatLLM] = None,
             default_input_chat_history: Optional[ChatHistory] = None,
             verbose: bool = False)
```

Initializes a ConditonalChatFlow.

**Arguments**:

- `decision_chat_flow`: Chat flow for making the decision.
- `branch_chat_flows`: Dictionary of chat flows for each branch. Use `default` as the key for the default branch.
- `default_chat_llm`: Optional default chat language model used in flow, if not provided in flow call.
- `default_input_chat_history`: Optional default input chat history used in flow, if not provided in flow call.
- `verbose`: If True, print chat flow messages.

<a id="flow.ConditonalChatFlow.verbose"></a>

#### verbose

```python
@property
def verbose()
```

**Returns**:

Whether the flow is verbose.

<a id="flow.ConditonalChatFlow.verbose"></a>

#### verbose

```python
@verbose.setter
def verbose(verbose: bool)
```

Sets the verbose attribute.

**Arguments**:

- `verbose`: Whether the flow is verbose.

<a id="flow.ConditonalChatFlow.flow"></a>

#### flow

```python
def flow(
    input_variables: dict,
    chat_llm: Optional[ChatLLM] = None,
    input_chat_history: Optional[ChatHistory] = None
) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]
```

Runs the decision chat flow through an LLM and then from the decision the appropriate branch.

**Arguments**:

- `input_variables`: Dictionary of input variables.
- `chat_llm`: Optional chat language model to use for the chat flow.
- `input_chat_history`: Optional input chat history.

**Returns**:

Tuple of dictionary of output variables and a tuple of input and internal chat histories.

<a id="flow.ConditonalChatFlow.compress_histories"></a>

#### compress\_histories

```python
def compress_histories(
    histories: Tuple[List[ChatHistory], List[ChatHistory]]
) -> Tuple[ChatHistory, ChatHistory]
```

Combines a tuple of list of chat histories into a tuple of two chat histories.

**Arguments**:

- `histories`: Tuple of list of chat histories.

**Returns**:

Tuple of combined input and internal chat histories.

<a id="flow.SequentialChatFlows"></a>

## SequentialChatFlows Objects

```python
class SequentialChatFlows(ChatFlowWrapper)
```

A sequential chat flow class that runs a list of chat flows sequentially.

Limitations:
 - All chat flows use the input history returned by the first chat flow plus internal of previous chat flows.
 - A chat flow can take an input and overwrite the original input with a new output with the same name. Be careful.

<a id="flow.SequentialChatFlows.__init__"></a>

#### \_\_init\_\_

```python
def __init__(chat_flows: List[ChatFlow],
             default_chat_llm: Optional[ChatLLM] = None,
             default_input_chat_history: Optional[ChatHistory] = None,
             verbose: bool = False) -> None
```

Initializes a seqeuntial chat flows class.

**Arguments**:

- `chat_flows`: List of chat flows to run sequentially.
- `default_chat_llm`: Optional default chat language model used in flow, if not provided in flow call.
- `default_input_chat_history`: Optional default input chat history used in flow, if not provided in flow call.
- `verbose`: If True, print chat flow messages.

<a id="flow.SequentialChatFlows.verbose"></a>

#### verbose

```python
@property
def verbose()
```

**Returns**:

Whether the flow is verbose.

<a id="flow.SequentialChatFlows.verbose"></a>

#### verbose

```python
@verbose.setter
def verbose(verbose: bool)
```

Sets the verbose attribute.

**Arguments**:

- `verbose`: Whether the flow is verbose.

<a id="flow.SequentialChatFlows.input_varnames"></a>

#### input\_varnames

```python
@property
def input_varnames()
```

**Returns**:

A deepcopy of input variable names.

<a id="flow.SequentialChatFlows.output_varnames"></a>

#### output\_varnames

```python
@property
def output_varnames()
```

**Returns**:

A deepcopy of output variable names.

<a id="flow.SequentialChatFlows.flow"></a>

#### flow

```python
def flow(
    input_variables: dict,
    chat_llm: Optional[ChatLLM] = None,
    input_chat_history: Optional[ChatHistory] = None
) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]
```

Runs the chat flows through an LLM.

**Arguments**:

- `input_variables`: Dictionary of input variables.
- `chat_llm`: Optional chat language model to use for the chat flow.
- `input_chat_history`: Optional input chat history.

**Returns**:

Tuple of dictionary of output variables and list of chat histories.

<a id="flow.ConcurrentChatFlows"></a>

## ConcurrentChatFlows Objects

```python
class ConcurrentChatFlows(ChatFlowWrapper)
```

<a id="flow.ConcurrentChatFlows.__init__"></a>

#### \_\_init\_\_

```python
def __init__(chat_flows: List[ChatFlow],
             default_chat_llm: Optional[ChatLLM] = None,
             default_input_chat_history: Optional[ChatHistory] = None,
             max_workers=None,
             verbose: bool = False) -> None
```

Initializes a concurrent chat flows class.

**Arguments**:

- `chat_flows`: List of chat flows to run concurrently.
- `default_chat_llm`: Optional default chat language model used in flow, if not provided in flow call.
- `default_input_chat_history`: Optional default input chat history used in flow, if not provided in flow call.
- `max_workers`: Number of threads to use for concurrent chat flows. If None, use all available threads.
- `verbose`: If True, print chat flow messages.

<a id="flow.ConcurrentChatFlows.verbose"></a>

#### verbose

```python
@property
def verbose()
```

**Returns**:

Whether the flow is verbose.

<a id="flow.ConcurrentChatFlows.verbose"></a>

#### verbose

```python
@verbose.setter
def verbose(verbose: bool)
```

Sets the verbose attribute.

**Arguments**:

- `verbose`: Whether the flow is verbose.

<a id="flow.ConcurrentChatFlows.input_varnames"></a>

#### input\_varnames

```python
@property
def input_varnames()
```

**Returns**:

A deepcopy of input variable names.

<a id="flow.ConcurrentChatFlows.output_varnames"></a>

#### output\_varnames

```python
@property
def output_varnames()
```

**Returns**:

A deepcopy of output variable names.

<a id="flow.ConcurrentChatFlows.flow"></a>

#### flow

```python
def flow(
    input_variables: dict,
    chat_llm: Optional[ChatLLM] = None,
    input_chat_history: Optional[ChatHistory] = None
) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]
```

Runs the chat flows concurrently through an LLM.

**Arguments**:

- `input_variables`: Dictionary of input variables.
- `chat_llm`: Optional chat language model to use for the chat flow.
- `input_chat_history`: Optional input chat history.

**Returns**:

Tuple of dictionary of output variables and tuple of list of chat histories (order matches ordering of chat_flows).

<a id="flow.ConcurrentChatFlows.compress_histories"></a>

#### compress\_histories

```python
def compress_histories(
    histories: Tuple[List[ChatHistory], List[ChatHistory]]
) -> Tuple[ChatHistory, ChatHistory]
```

Combines a tuple of list of chat histories into a tuple of two chat histories.

**Arguments**:

- `histories`: Tuple of list of chat histories.

**Returns**:

Tuple of combined input and internal chat histories.

<a id="flow.ChatSpiral"></a>

## ChatSpiral Objects

```python
class ChatSpiral(ChatFlowWrapper)
```

<a id="flow.ChatSpiral.__init__"></a>

#### \_\_init\_\_

```python
def __init__(chat_flow: ChatFlow,
             output_varnames_remap: Optional[Dict[str, str]] = None,
             default_chat_llm: Optional[ChatLLM] = None,
             default_input_chat_history: Optional[ChatHistory] = None,
             verbose: bool = False) -> None
```

Initializes a chat spiral class.

**Arguments**:

- `chat_flow`: Chat flow to spiral.
- `output_varnames_remap`: Optional dictionary of output variable names to remap.
- `default_chat_llm`: Optional default chat language model used in flow, if not provided in flow/spiral call.
- `default_input_chat_history`: Optional default input chat history used in flow, if not provided in flow/spiral call.

<a id="flow.ChatSpiral.flow"></a>

#### flow

```python
def flow(
    input_variables: dict,
    chat_llm: Optional[ChatLLM] = None,
    input_chat_history: Optional[ChatHistory] = None
) -> Tuple[Dict[str, str], Tuple[List[ChatHistory], List[ChatHistory]]]
```

Runs the chat flow through an LLM.

**Arguments**:

- `input_variables`: Dictionary of input variables.
- `chat_llm`: Optional chat language model to use for the chat flow.
- `input_chat_history`: Optional input chat history.

**Returns**:

Tuple of dictionary of output variables and two tuple of list of chat histories.

<a id="flow.ChatSpiral.compress_histories"></a>

#### compress\_histories

```python
def compress_histories(
    histories: Tuple[List[ChatHistory], List[ChatHistory]]
) -> Tuple[ChatHistory, ChatHistory]
```

Combines a tuple of list of chat histories into a tuple of two chat histories.

**Arguments**:

- `histories`: Tuple of list of chat histories.

**Returns**:

Tuple of combined input and internal chat histories.

<a id="memory"></a>

# memory

<a id="memory.Memory"></a>

## Memory Objects

```python
class Memory()
```

<a id="memory.Memory.__init__"></a>

#### \_\_init\_\_

```python
def __init__(filepath: Optional[str] = None,
             embedding_model: str = "text-embedding-ada-002",
             max_tokens: int = 500) -> None
```

Initializes the memory.

**Arguments**:

- `filepath`: Path to a pickle file to load and save the memory to.
If None, the memory is created with text and metadata fields.
- `embedding_model`: Model to use for the embedding.
- `max_tokens`: Maximum number of tokens to use for the embedding.

<a id="memory.Memory.save"></a>

#### save

```python
def save(filepath: Optional[str] = None) -> None
```

Saves the memory to a file.

**Arguments**:

- `filepath`: Path to the pickle file to save the memory to. If None, the filepath passed in the constructor is used.

<a id="memory.Memory.load"></a>

#### load

```python
def load(filepath: Optional[str] = None) -> None
```

**Arguments**:

- `filepath`: Path to a pickle file to load the memory from. If None, the filepath passed in the constructor is used.

<a id="memory.Memory.add"></a>

#### add

```python
def add(data: Dict[str, str],
        save: bool = False,
        filepath: Optional[str] = None) -> None
```

Adds data to memory.

**Arguments**:

- `data`: Dict of data with a text and metadata field to add to memory.
- `save`: Whether to save the memory to a file.
- `filepath`: Path to the file (csv or parquet) to save the memory to.
If None, the filepath passed in the constructor is used.

<a id="memory.Memory.query"></a>

#### query

```python
def query(query: str, k: int = 1) -> list[Dict[str, str]]
```

Queries the memory with the given query.

**Arguments**:

- `query`: Query to use to get memory.
- `k`: Number of results to return.

**Returns**:

Memory obtained from external memories.

<a id="message"></a>

# message

<a id="message.extract_fstring_variables"></a>

#### extract\_fstring\_variables

```python
def extract_fstring_variables(text: str) -> List[str]
```

Extracts variables from a f-string like text.

**Arguments**:

- `text`: f-string like text to extract variables from.

<a id="message.flatten_dict"></a>

#### flatten\_dict

```python
def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict
```

Flatten a dictionary.

**Arguments**:

- `d`: Dictionary to flatten.
- `parent_key`: Parent key to use.
- `sep`: Separator to use.

**Returns**:

Flattened dictionary.

<a id="message.ExtractionError"></a>

## ExtractionError Objects

```python
class ExtractionError(Exception)
```

A class to represent an error in extracting a variable from a message.

<a id="message.Role"></a>

## Role Objects

```python
class Role()
```

A class to represent the role of a message. Using OpenAI roles.

<a id="message.Message"></a>

## Message Objects

```python
class Message(ABC)
```

A class to represent a message.

<a id="message.Message.__init__"></a>

#### \_\_init\_\_

```python
def __init__(content_format: str, role: Optional[str] = None) -> None
```

Initializes the Message class with the given parameters.

**Arguments**:

- `content_format`: A f-string format for the message content.
- `role`: Role associated with the message (default is None).

<a id="message.Message.content_format"></a>

#### content\_format

```python
@content_format.setter
def content_format(content_format: str)
```

**Arguments**:

- `content_format`: A f-string like format for the message content.

<a id="message.Message.role"></a>

#### role

```python
@role.setter
def role(role: str)
```

**Arguments**:

- `role`: Role associated with the message.

<a id="message.Message.defined"></a>

#### defined

```python
def defined() -> bool
```

Determines if all variables have a value, essentially if the message has been called or has no variables.

**Returns**:

True if all variables have a value, False otherwise.

<a id="message.Message.make_const"></a>

#### make\_const

```python
def make_const() -> None
```

Makes this message constant so variables and content format cannot change.

<a id="message.Message.get_const"></a>

#### get\_const

```python
def get_const() -> Any
```

Creates a deepcopy of self and makes it constant.

**Returns**:

A deepcopy of this message made constant so variables and content format cannot change.

<a id="message.Message.__call__"></a>

#### \_\_call\_\_

```python
@abstractmethod
def __call__(**kwargs: Any) -> Any
```

A method to run content through to get variables or to put variables in to form a content.

<a id="message.Message.__str__"></a>

#### \_\_str\_\_

```python
def __str__() -> str
```

**Returns**:

The message content if defined, otherwise the message content format.

<a id="message.InputMessage"></a>

## InputMessage Objects

```python
class InputMessage(Message)
```

A class to represent a message that takes variables as inputs to construct.

<a id="message.InputMessage.__init__"></a>

#### \_\_init\_\_

```python
def __init__(content_format: str,
             role: Optional[str] = Role.USER,
             custom_insert_variables_func: Optional[Callable[[Dict[str, Any]],
                                                             str]] = None)
```

Initializes the InputMessage class with the given parameters.

**Arguments**:

- `content_format`: A f-string format for the message content.
- `role`: Role associated with the message (default is None).
- `custom_insert_variables_func`: A custom function to insert variables into the message content.
Takes the content_format and a dictionary of variables and returns the message content.

<a id="message.InputMessage.__call__"></a>

#### \_\_call\_\_

```python
def __call__(**kwargs: Any) -> str
```

Get the message content with inserted variables.

**Arguments**:

- `kwargs`: A dictionary containing variable values.

**Returns**:

The message content with inserted variables.

<a id="message.InputMessage.insert_variables"></a>

#### insert\_variables

```python
def insert_variables(variables: Dict[str, Any]) -> str
```

Insert variables into the message content.

**Arguments**:

- `variables`: A dictionary containing variable values.

**Returns**:

The message content with inserted variables.

<a id="message.OutputMessage"></a>

## OutputMessage Objects

```python
class OutputMessage(Message)
```

A class to represent a message that outputs variables from its message content.

Limitations:
- Variables must be seperated. Regex pattern used: (?P<{}>[\s\S]*)

<a id="message.OutputMessage.__init__"></a>

#### \_\_init\_\_

```python
def __init__(
    content_format: str,
    role: Optional[str] = Role.ASSISTANT,
    custom_extract_variables_func: Optional[Callable[[List[str], str, str],
                                                     Dict[str, Any]]] = None)
```

Initializes the OutputMessage class with the given parameters.

**Arguments**:

- `content_format`: A f-string format for the message content.
- `role`: Role associated with the message (default is None).
- `custom_extract_variables_func`: A custom function to extract variables from the message content.
Takes a list of variable names, the content format, and the message content.
Returns a dictionary containing the extracted variables.

<a id="message.OutputMessage.__call__"></a>

#### \_\_call\_\_

```python
def __call__(**kwargs: Any) -> Dict[str, Any]
```

Extract variables from the message content.

**Arguments**:

- `kwargs`: A dictionary containing the message content.

**Returns**:

A dictionary containing the extracted variables.

<a id="message.OutputMessage.extract_variables"></a>

#### extract\_variables

```python
def extract_variables(content) -> Dict[str, Any]
```

Extract variables from the message content.

**Arguments**:

- `content`: The message content to extract variables from.

**Returns**:

A dictionary containing the extracted variables.

<a id="message.InputJSONMessage"></a>

## InputJSONMessage Objects

```python
class InputJSONMessage(InputMessage)
```

A class to represent a message that takes JSON dict keys-values as inputs to construct.

Limitations:
- Sub-dictionaries are accessed by periods and replaced with underscores in processing, so name conflicts can occur.

<a id="message.InputJSONMessage.__init__"></a>

#### \_\_init\_\_

```python
def __init__(content_format: str,
             role: Optional[str] = Role.USER,
             expected_input_varnames: Optional[Set[str]] = None)
```

Initializes the InputJSONMessage class with the given parameters.

**Arguments**:

- `content_format`: A f-string format for the message content.
- `role`: Role associated with the message (default is None).
- `expected_input_varnames`: A set of expected input variable names.

<a id="message.InputJSONMessage.insert_variables_into_json"></a>

#### insert\_variables\_into\_json

```python
def insert_variables_into_json(content_format: str,
                               variables: Dict[str, Any]) -> str
```

Insert variables from dict into the message content.

**Arguments**:

- `content_format`: The message content format.
- `variables`: A dictionary containing variable values.

**Returns**:

The message content with inserted variables.

<a id="message.OutputJSONMessage"></a>

## OutputJSONMessage Objects

```python
class OutputJSONMessage(OutputMessage)
```

A class to represent a message that outputs JSON dict keys-values from its message content.

Limitations:
- Only supports JSON dicts as outputs.
- Regex patterns do not necessarily match every content_format possible.

<a id="message.OutputJSONMessage.__init__"></a>

#### \_\_init\_\_

```python
def __init__(content_format: str, role: Optional[str] = Role.ASSISTANT)
```

Initializes the OutputJSONMessage class with the given parameters.

**Arguments**:

- `content_format`: A f-string format for the message content.
- `role`: Role associated with the message (default is None).

<a id="message.OutputJSONMessage.extract_variables_from_json"></a>

#### extract\_variables\_from\_json

```python
def extract_variables_from_json(names: List[str], content_format: str,
                                content: str) -> Dict[str, Any]
```

Extract JSON Dict from the message content.

**Arguments**:

- `names`: A list of variable names.
- `content_format`: The message content format.
- `content`: The message content to extract variables from.

**Returns**:

A dictionary containing the extracted variables.

<a id="tools"></a>

# tools

<a id="tools.GoogleSearchTool"></a>

## GoogleSearchTool Objects

```python
class GoogleSearchTool(BaseTool)
```

<a id="tools.GoogleSearchTool.__init__"></a>

#### \_\_init\_\_

```python
def __init__(api_key: str,
             cse_id: str,
             num_results: int = 10,
             failed_search_result: str = "No google search results found.",
             join_snippets: Optional[str] = "\n") -> None
```

Initialize the GoogleSearchTool.

**Arguments**:

- `api_key`: The Google API key.
- `cse_id`: The Google Custom Search Engine ID.
- `num_results`: The max number of results to return.
- `failed_search_result`: The result to return if the search fails.
- `join_snippets`: The string to join the snippets with. If None, the snippets will be returned as a list.

<a id="tools.GoogleSearchTool.search"></a>

#### search

```python
def search(query: str) -> Optional[list]
```

**Arguments**:

- `query`: The query to search for.

**Returns**:

The search results.

<a id="tools.GoogleSearchTool.use"></a>

#### use

```python
def use(inputs: Dict[str, str]) -> Union[str, list[str]]
```

**Arguments**:

- `inputs`: The inputs to the tool. Must contain a 'query' key.

**Returns**:

The output of the tool: Google search snippets.

<a id="__init__"></a>

# \_\_init\_\_

