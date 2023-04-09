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
from message import Role, InputMessage, OutputMessage, InputJSONMessage, OutputJSONMessage
from chat_llm import ChatLLM
from chat_history import ChatHistory
from chat_flow import ChatFlow

# Create input and output messages
input_system_msg = InputMessage('All your outputs follow the format: The capital of country is capital country.', role=Role.SYSTEM)

input_msg1 = InputMessage('What is the capital of {country1}?')
output_msg1 = OutputMessage('The capital of {country1} is {capital1}.')

input_msg2 = InputMessage('What is the capital of {country2}?')
output_msg2 = OutputMessage('The capital of {country2} is {capital2}.')

input_system_msg2 = InputMessage('You are a historian who writes very brief comparison analyses.', role=Role.SYSTEM)

input_msg3 = InputMessage(
    'Compare {capital1} and {capital2}? Ensure the response can be parsed by Python json.loads exactly as {{"analyses": {{"a": "details", "b": "details"}}}}')
output_msg3 = OutputJSONMessage('{comparison}')

input_msg4 = InputJSONMessage('Is the following that you said true:\n{comparison.analyses.a}')
output_msg4 = OutputMessage('{reflect_response}')

# Initialize the ChatLLM with the desired OpenAI model
chat_llm = ChatLLM(gpt_model='gpt-3.5-turbo')

# Create a ChatFlow with the input and output messages
chat_flow = ChatFlow([input_system_msg, input_msg1, output_msg1, input_msg2, output_msg2,
                      input_system_msg2, input_msg3, output_msg3,
                      input_msg4, output_msg4])

# Define input variables for the ChatFlow
input_variables = {'country1': 'France', 'country2': 'Spain'}

# Execute the ChatFlow and obtain the extracted variables and chat history
variables, internal_chat_history = chat_flow(chat_llm, input_variables)

print('Extracted Variables:', variables)
print('Chat History:')
for message in internal_chat_history.messages:
    print(f'{message.role.title()}: {message.content}')
```

This will output:

```
Extracted Variables: {'reflect_response': 'Yes, that is true. Paris is located in the northern part of France, while Madrid is located in the central part of Spain. The population of Paris is around 2.2 million, and the population of Madrid is around 3.3 million. Both cities have a rich culture, with numerous museums, galleries, and other cultural attractions. Paris is known for its landmarks like the Eiffel Tower and the Louvre museum, while Madrid is famous for its nightlife and historic buildings such as the Royal Palace and Plaza Mayor.'}
Chat History:
System: All your outputs follow the format: The capital of country is capital country.
User: What is the capital of France?
Assistant: The capital of France is Paris.
User: What is the capital of Spain?
Assistant: The capital of Spain is Madrid.
System: You are a historian who writes very brief comparison analyses.
User: Compare Paris and Madrid? Ensure the response can be parsed by Python json.loads exactly as {"analyses": {"a": "details", "b": "details"}}
Assistant: {"analyses": {
    "a": "Paris is located in northern France while Madrid is located in central Spain. The populations of the two cities are similar, with around 2.2 million people living in Paris and around 3.3 million people living in Madrid. Both cities are home to many museums, galleries, and other cultural attractions. Paris is known for its iconic landmarks such as the Eiffel Tower and the Louvre museum, while Madrid is known for its vibrant nightlife and its many historic buildings such as the Royal Palace and Plaza Mayor.",
    "b": "Paris and Madrid both have extensive public transportation systems, with Paris having the famous metro system and Madrid having an extensive bus and metro network. Both cities are also major transportation hubs, with multiple airports and high-speed train connections to other cities throughout Europe. However, Paris is generally considered to be more tourist-oriented than Madrid, with a larger and more developed tourism industry. Finally, both Paris and Madrid are known for their food and wine cultures, with many great restaurants and cafes serving delicious cuisine." 
}}
User: Is the following that you said true:
Paris is located in northern France while Madrid is located in central Spain. The populations of the two cities are similar, with around 2.2 million people living in Paris and around 3.3 million people living in Madrid. Both cities are home to many museums, galleries, and other cultural attractions. Paris is known for its iconic landmarks such as the Eiffel Tower and the Louvre museum, while Madrid is known for its vibrant nightlife and its many historic buildings such as the Royal Palace and Plaza Mayor.
Assistant: Yes, that is true. Paris is located in the northern part of France, while Madrid is located in the central part of Spain. The population of Paris is around 2.2 million, and the population of Madrid is around 3.3 million. Both cities have a rich culture, with numerous museums, galleries, and other cultural attractions. Paris is known for its landmarks like the Eiffel Tower and the Louvre museum, while Madrid is famous for its nightlife and historic buildings such as the Royal Palace and Plaza Mayor.
```

License
-------

This project is released under the MIT License.
