# spiral (Work-In-Progress)
A framework for creating guided spirals for Large Language Models

This project is designed to help with creating, formatting, and extracting data from text-based conversations with OpenAI language models to allow for more complicated ideas such as Flows and Spirals. It includes the following key components:

*   `Message`: A base class representing a message with content, role, and variables.
*   `InputMessage`: A class to create input messages with pre-defined content formats.
*   `OutputMessage`: A class to create output messages and extract variables from them.
*   `InputJSONMessage`: A class to create input messages with pre-defined content formats that are able to access Dict variables.
*   `OutputJSONMessage`: A class to create output messages and extract JSON formated dict variables from them.
*   `ChatLLM`: A class to interface with OpenAI's chat models.
*   `ChatHistory`: A class to store a sequence of messages in a conversation.
*   `ChatFlow`: A class to represent a conversation flow, combining input and output messages.
*   `ChatFlows`: A class to represent ultiple conversation flows, flowing sequentially.
*   `ConditonalChatFlow`: A class to represent a conversation flow with multiple branches that depend on an output of a flow.
*   `ChatFlowManager`: A class to manage multiple chat flows with their llms and conversation histories.

Installation
------------

To use this project, you will need to install the required dependencies:

1.  Install Python 3.8 or higher.
2.  Install the `openai` package: `pip install openai`.
3.  Install the `regex` package: `pip install regex`.

Usage
-----

Here is a quick example to demonstrate how to use the project:

```python
from message import (
    Role,
    InputMessage,
    OutputMessage,
    InputJSONMessage,
    OutputJSONMessage,
)
from chat_llm import ChatLLM
from chat_history import ChatHistory
from flow import ChatFlow


# Create input and output messages
chat_flow = ChatFlow.from_dicts([
    {Role.SYSTEM: 'All your outputs follow the format: The capital of country is capital country.'},
    {Role.USER: 'What is the capital of {country1}?'},
    {Role.ASSISTANT: 'The capital of {country1} is {capital1}.', 'type': 'output'},  # optionally specify the type of message for USER/ASSISTANT
    {Role.USER: 'What is the capital of {country2}?'},
    {Role.ASSISTANT: 'The capital of {country2} is {capital2}.'},
    {Role.SYSTEM: 'You are a historian who writes very brief comparison analyses.'},
    {Role.USER: 'Compare {capital1} and {capital2}? Ensure the response can be parsed by Python json.loads exactly as {{"analyses": {{"a": "details", "b": "details"}}}}'},
    {Role.ASSISTANT: '{comparison}', 'type': 'output_json'},
    {Role.USER: 'Is the following you said true for {capital1}, {country1}:\n{comparison.analyses.a}', 'type': 'input_json', 'expected_input_varnames': {'comparison.analyses.a'}},
    {Role.ASSISTANT: '{reflect_response}'}
], verbose=False)

# Initialize the ChatLLM with the desired OpenAI model
chat_llm = ChatLLM(gpt_model="gpt-3.5-turbo", temperature=0.3)

# Define input variables for the ChatFlow
input_variables = {"country1": "France", "country2": "Spain"}

# Starting conversation
input_chat_history = ChatHistory(
    [
        InputMessage("What is the capital of Mexico?"),
        OutputMessage("The capital of Mexico is Mexico City."),
        InputMessage("What is the capital of United States?"),
        OutputMessage("The capital of United States is Washington, D.C."),
    ]
)

# Execute the ChatFlow and obtain the extracted variables and chat history
variables, history = chat_flow(
    input_variables, chat_llm=chat_llm, input_chat_history=input_chat_history
)

print("Extracted Variables:", variables)
print("Chat History:")
for message in history.messages:
    print(f"{message.role.title()}: {message.content}")
```

This will output:

```
Extracted Variables: {'reflect_response': 'Yes, that statement is true. Paris is the capital of France, while Madrid is the capital of Spain.'}
Chat History:
User: What is the capital of Mexico?
Assistant: The capital of Mexico is Mexico City.
User: What is the capital of United States?
Assistant: The capital of United States is Washington, D.C.
System: All your outputs follow the format: The capital of country is capital country.
User: What is the capital of France?
Assistant: The capital of France is Paris.
User: What is the capital of Spain?
Assistant: The capital of Spain is Madrid.
System: You are a historian who writes very brief comparison analyses.
User: Compare Paris and Madrid? Ensure the response can be parsed by Python json.loads exactly as {"analyses": {"a": "details", "b": "details"}}
Assistant: {"analyses": {"a": "Paris is the capital of France, while Madrid is the capital of Spain.", "b": "Paris is known for its art, fashion, and cuisine, while Madrid is known for its museums, parks, and nightlife."}}
User: Is the following you said true for Paris, France:
Paris is the capital of France, while Madrid is the capital of Spain.
Assistant: Yes, that statement is true. Paris is the capital of France, while Madrid is the capital of Spain.
```

License
-------

This project is released under the MIT License.
