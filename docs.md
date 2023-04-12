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

<a id="chat_history.ChatHistory.add_message"></a>

#### add\_message

```python
def add_message(message: Message) -> None
```

Adds a message made constant to the chat history.

**Arguments**:

- `message`: The message to add to the chat history.

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
def __init__(gpt_model: str = 'gpt-3.5-turbo', stream=False, **kwargs) -> None
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
             default_input_chat_history: Optional[ChatHistory] = None,
             verbose: bool = False) -> None
```

Initializes the ChatFlow class with the given parameters.

**Arguments**:

- `messages`: List of messages in the chat flow.
- `default_input_chat_history`: Optional default input chat history used in flow, if not provided in flow call.
- `verbose`: Whether to print verbose output.

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
    chat_llm: ChatLLM,
    input_variables: dict,
    input_chat_history: Optional[ChatHistory] = None
) -> Tuple[Dict[str, str], List[ChatHistory]]
```

Runs the chat flow through an LLM.

**Arguments**:

- `chat_llm`: The chat language model to use for the chat flow.
- `input_variables`: Dictionary of input variables.
- `input_chat_history`: Optional input chat history.

**Returns**:

Tuple of dictionary of output variables and chat histories.

<a id="flow.ChatFlow.__call__"></a>

#### \_\_call\_\_

```python
def __call__(chat_llm: ChatLLM,
             input_variables: dict,
             input_chat_history: Optional[ChatHistory] = None,
             return_all: bool = True) -> Tuple[Dict[str, str], ChatHistory]
```

Runs the chat flow through an LLM.

**Arguments**:

- `chat_llm`: The chat language model to use for the chat flow.
- `input_variables`: Dictionary of input variables.
- `input_chat_history`: Optional input chat history.
- `return_all`: If True, return all variables. If False, return only output variables.

**Returns**:

Tuple of dictionary of output variables and chat history.

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
def __init__(func: Callable[[ChatLLM, dict, Optional[ChatHistory]],
                            Tuple[Dict[str, str], List[ChatHistory]]],
             input_varnames: Set[str],
             output_varnames: Set[str],
             default_input_chat_history: Optional[ChatHistory] = None,
             verbose: bool = False) -> None
```

Initializes a FuncChatFlow.

**Arguments**:

- `func`: Function to use for the chat flow.
- `input_varnames`: List of input variable names.
- `output_varnames`: List of output variable names.
- `default_input_chat_history`: Optional default input chat history used in flow, if not provided in
- `verbose`: If True, print chat flow steps.

<a id="flow.FuncChatFlow.flow"></a>

#### flow

```python
def flow(
    chat_llm: ChatLLM,
    input_variables: dict,
    input_chat_history: Optional[ChatHistory] = None
) -> Tuple[Dict[str, str], List[ChatHistory]]
```

Runs the chat flow through an LLM.

**Arguments**:

- `chat_llm`: The chat language model to use for the chat flow.
- `input_variables`: Dictionary of input variables.
- `input_chat_history`: Optional input chat history.

**Returns**:

Tuple of dictionary of output variables and chat histories.

<a id="flow.ConditonalChatFlow"></a>

## ConditonalChatFlow Objects

```python
class ConditonalChatFlow(ChatFlow)
```

A class for creating conditional chat flows, which shift flows based on the output of previous messages.

<a id="flow.ConditonalChatFlow.__init__"></a>

#### \_\_init\_\_

```python
def __init__(decision_chat_flow: ChatFlow,
             branch_chat_flows: Dict[str, ChatFlow],
             share_input_history: bool = True,
             share_internal_history: bool = True,
             default_input_chat_history: Optional[ChatHistory] = None,
             verbose: bool = False)
```

Initializes a ConditonalChatFlow.

**Arguments**:

- `decision_chat_flow`: Chat flow for making the decision.
- `branch_chat_flows`: Dictionary of chat flows for each branch.
- `share_input_history`: If True, share the input chat history between the decision and branch chat flows.
- `share_internal_history`: If True, share the internal chat history between the decision and branch chat flows.
- `default_input_chat_history`: Optional default input chat history used in flow, if not provided in flow call.
- `verbose`: If True, print chat flow messages.

<a id="flow.ConditonalChatFlow.flow"></a>

#### flow

```python
def flow(
    chat_llm: ChatLLM,
    input_variables: dict,
    input_chat_history: Optional[ChatHistory] = None
) -> Tuple[Dict[str, str], List[ChatHistory]]
```

Runs the decision chat flow through an LLM and then from the decision the appropriate branch.

**Arguments**:

- `chat_llm`: The chat language model to use for the chat flow.
- `input_variables`: Dictionary of input variables.
- `input_chat_history`: Optional input chat history.

**Returns**:

Tuple of dictionary of output variables and list of chat histories.

<a id="flow.ChatFlows"></a>

## ChatFlows Objects

```python
class ChatFlows(ChatFlow)
```

<a id="flow.ChatFlows.flow"></a>

#### flow

```python
def flow(
    chat_llm: ChatLLM,
    input_variables: dict,
    input_chat_history: Optional[ChatHistory] = None
) -> Tuple[Dict[str, str], List[ChatHistory]]
```

Runs the chat flows through an LLM.

**Arguments**:

- `chat_llm`: The chat language model to use for the chat flow.
- `input_variables`: Dictionary of input variables.
- `input_chat_history`: Optional input chat history.

**Returns**:

Tuple of dictionary of output variables and list of chat histories.

<a id="flow.ChatFlowManager"></a>

## ChatFlowManager Objects

```python
class ChatFlowManager()
```

A class for managing chat flows.

<a id="flow.ChatFlowManager.ChatFlowNode"></a>

## ChatFlowNode Objects

```python
class ChatFlowNode()
```

A class for adding mapping information to a chat flow.

<a id="flow.ChatFlowManager.ChatFlowNode.__init__"></a>

#### \_\_init\_\_

```python
def __init__(name,
             chat_flow: ChatFlow,
             input_varnames_map: Optional[Dict[str, str]] = None,
             output_varnames_map: Optional[Dict[str, str]] = None,
             input_chat_history_map: Optional[List[Dict[str, Any]]] = None)
```

Initializes a ChatFlowNode.

**Arguments**:

- `name`: Name of the chat flow.
- `chat_flow`: Chat flow.
- `input_varnames_map`: Optional dictionary of input variable names to map to the chat flow.
- `output_varnames_map`: Optional dictionary of output variable names to map to the chat flow.
- `input_chat_history_map`: Optional list of dictionaries of input chat history metadata to map to the chat flow.

<a id="flow.ChatFlowManager.ChatFlowNode.flow"></a>

#### flow

```python
def flow(
    chat_llm: ChatLLM, input_variables: dict,
    input_chat_histories: Dict[str, Dict[str, ChatHistory]]
) -> Tuple[Dict[str, str], List[ChatHistory]]
```

Runs the chat flow through an LLM while remapping input variables, chat histories, and output variables.

**Arguments**:

- `chat_llm`: The chat language model to use for the chat flow.
- `input_variables`: Dictionary of input variables.
- `input_chat_histories`: Dictionary of input chat histories.

**Returns**:

Tuple of dictionary of output variables and chat histories.

<a id="flow.ChatFlowManager.__init__"></a>

#### \_\_init\_\_

```python
def __init__(chat_flow_nodes: List[ChatFlowNode])
```

Initializes a ChatFlowManager.

**Arguments**:

- `chat_flow_nodes`: List of ChatFlowNodes in sequential order.

<a id="flow.ChatFlowManager.flow"></a>

#### flow

```python
def flow(
    chat_llms: Dict[str, ChatLLM], input_variables: dict,
    input_chat_histories: Dict[str, ChatHistory]
) -> Tuple[Dict[str, str], List[ChatHistory]]
```

Runs all the chat flows through the LLMs while remapping input variables, chat histories, and output variables.

**Arguments**:

- `chat_llms`: Dictionary of chat language models with names mapping to chat flow node names.
- `input_variables`: Dictionary of input variables.
- `input_chat_histories`: Dictionary of input chat histories with names (do not need to be same as chat flow node names)

**Returns**:

Tuple of dictionary of output variables and chat histories.

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
def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict
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
