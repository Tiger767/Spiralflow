# spiral (Work-In-Progress)
A framework for creating guided spirals for Large Language Models

This project is designed to help with creating, formatting, and extracting data from text-based conversations with OpenAI language models. It includes the following key components:

*   `Message`: A base class representing a message with content, role, and variables.
*   `InputMessage`: A class to create input messages with pre-defined content formats.
*   `OutputMessage`: A class to create output messages and extract variables from them.
*   `ChatLLM`: A class to interface with OpenAI's chat models.
*   `ChatHistory`: A class to store a sequence of messages in a conversation.
*   `ChatFlow`: A class to represent a conversation flow, combining input and output messages.

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
from message import InputMessage, OutputMessage, Role
from chat_llm import ChatLLM
from chat_history import ChatHistory
from chat_flow import ChatFlow

# Create input and output messages
input_system_msg = InputMessage("All your outputs follow the format: The capital of country is capital country.", role=Role.SYSTEM)
input_msg = InputMessage("What is the capital of {country}?")
output_msg = OutputMessage("The capital of {country} is {capital}.")

# Initialize the ChatLLM with the desired OpenAI model
chat_llm = ChatLLM(gpt_model="gpt-3.5-turbo")

# Create a ChatFlow with the input and output messages
chat_flow = ChatFlow([input_system_msg, input_msg, output_msg])

# Define input variables for the ChatFlow
input_variables = {"country": "France"}

# Execute the ChatFlow and obtain the extracted variables and chat history
variables, internal_chat_history = chat_flow(chat_llm, input_variables)

print("Extracted Variables:", variables)
print("Chat History:")
for message in internal_chat_history.messages:
    print(message.role, message.content)
```

This will output:

```
Extracted Variables: {'country': 'France', 'capital': 'Paris'}
Chat History:
user What is the capital of France?
assistant The capital of France is Paris.
```

License
-------

This project is released under the MIT License.
