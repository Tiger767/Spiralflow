# spiral-flow (Work-In-Progress)
A framework for creating guided spirals for Large Language Models

This project is designed to help with creating, formatting, and extracting data from text-based conversations with OpenAI language models to allow for more complicated ideas such as Flows and Spirals. It includes the following key components:

*   `Message`: A base class representing a message with content, role, and variables.
*   `InputMessage`: A class to create input messages with pre-defined content formats.
*   `OutputMessage`: A class to create output messages and extract variables from them.
*   `InputJSONMessage`: A class to create input messages with pre-defined content formats that are able to access Dict variables.
*   `OutputJSONMessage`: A class to create output messages and extract JSON formated dict variables from them.
*   `ChatLLM`: A class to interface with OpenAI's chat models.
*   `ChatHistory`: A class to store a sequence of messages in a conversation.
*   `ChatHistoryManager`: A class to manage many different chat histories.
*   `ChatFlow`: A class to represent a conversation flow, combining input and output messages.
*   `FuncChatFlow`: A class to represent a function in a conversation flow.
*   `ChatFlowWrapper`: A class to wrap ChatFlows to allow for more complicated ideas such as NoHistory.
*   `NoHistory`: A class to block the flow of specific histories whether in or out of the flow.
*   `History`: A class that allows for more complicated flows of histories, utilizing the ChatHistoryManager.
*   `MemoryChatFlow`: A class that allows a chat flow to query external memory.
*   `ConditonalChatFlow`: A class to represent a conversation flow with multiple branches that depend on an output of a flow.
*   `SequentialChatFlows`: A class to represent multiple conversation flows, flowing sequentially.
*   `ConcurrentChatFlows`: A class to represent multiple conversation flows, flowing separately and concurrently.
*   `ChatSpiral`: A class to represent a spiral of conversation flows.


Installation
------------

To use this project, you will need to install the required dependencies:

1.  Install Python 3.9 or higher.
2.  Install the `openai`, `tiktoken`, `pandas`, packages: `pip install openai tiktoken pandas`.
3.  Install faiss package: `conda install -c conda-forge pytorch faiss-cpu`
3.  Install the this package: `git clone https://github.com/Tiger767/spiral.git` and then `pip install .` inside the repo.
4.  Make sure OPENAI_API_KEY is set as an environment variable with your OpenAI API key.

Usage
-----

Here is a quick example to demonstrate how to use the project:

```python
from spiralflow.message import (
    Role,
    InputMessage,
    OutputMessage,
    InputJSONMessage,
    OutputJSONMessage,
)
from spiralflow.chat_llm import ChatLLM
from spiralflow.chat_history import ChatHistory
from spiralflow.flow import ChatFlow


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

Here is an example using a *spiral* and some of the more advanced features:

```python
from spiralflow.message import Role
from spiralflow.chat_llm import ChatLLM
from spiralflow.flow import ChatFlow, ConditonalChatFlow, FuncChatFlow, NoHistory, ChatSpiral

decision_flow = ChatFlow.from_dicts([
    {Role.USER: 'Options:\n\n'
                '1. Keep asking and answering questions.\n\n'
                '2. Choose this when you have provided 2 prompts in the total conversation history.\n\n'
                'Only respond by typing the number of the option you choose. Say nothing else other than `1` or `2`.'},
    {Role.ASSISTANT: '{decision}'}
], verbose=True)

branch1_flow = ChatFlow.from_dicts([
    {Role.USER: 'Prompt: {prompt}\n\n'},
    {Role.ASSISTANT: '{answer}'},
    {Role.USER: 'Give me a new prompt.'},
    {Role.ASSISTANT: '{prompt}'},
], verbose=True)

def quit(variables, chat_llm, input_chat_history):
    raise ChatSpiral.Exit()

branch2_flow = FuncChatFlow(
    func=quit, input_varnames=set([]), output_varnames=set(['answer', 'prompt']), verbose=True
)

cond_flow = ConditonalChatFlow(NoHistory(decision_flow, allow_input_history=True, allow_rtn_input_history=True),
                               {'1': branch1_flow, '2': branch2_flow}, verbose=True)

a_spiral = ChatSpiral(cond_flow, verbose=False)

chat_llm = ChatLLM(gpt_model="gpt-3.5-turbo", temperature=0.3)

variables, history = a_spiral.spiral({'prompt': 'What is math?'}, chat_llm=chat_llm, max_iterations=10, reset_history=False)

print("Extracted Variables:")
for key, value in variables.items():
    print(f"{key}: {value}\n")
print("Chat History:")
for message in history.messages:
    print(f"{message.role.title()}: {message.content}")
```

This will output something like:

```
Extracted Variables:
prompt: Sure, here's your new prompt:

Explain the greenhouse effect and its impact on climate change.

decision: 1

answer: Artificial intelligence (AI) is a branch of computer science that deals with the creation of intelligent machines that can perform tasks that typically require human intelligence. AI involves the development of algorithms and computer programs that can learn from data, recognize patterns, and make decisions based on that information. 

There are two main types of AI: narrow or weak AI and general or strong AI. Narrow AI is designed to perform a specific task, such as playing chess or driving a car. It is programmed to follow a set of rules and make decisions based on those rules. General AI, on the other hand, is designed to be more flexible and adaptable. It can learn and reason like a human, and can perform a wide range of tasks across different domains.

AI is used in a variety of applications, from virtual assistants like Siri and Alexa to self-driving cars and medical diagnosis. It has the potential to revolutionize many industries and improve our lives in countless ways. However, there are also concerns about the ethical implications of AI, such as the potential for job displacement and the risk of bias in decision-making algorithms. As AI continues to evolve and become more advanced, it will be important to carefully consider these issues and ensure that the technology is used in a responsible and beneficial way.

Chat History:
User: Prompt: What is math?


Assistant: Math, short for mathematics, is a subject that deals with the study of numbers, quantities, and shapes. It involves the use of logical reasoning and critical thinking to solve problems and make sense of the world around us. Math is a fundamental tool in many fields, including science, engineering, finance, and technology. It encompasses various branches such as algebra, geometry, calculus, statistics, and more. Math is not just about memorizing formulas and equations, but also about understanding concepts and applying them to real-world situations. It plays a crucial role in our daily lives, from calculating the cost of groceries to designing complex structures. Overall, math is a universal language that helps us make sense of the world and solve problems in a logical and systematic way.
User: Give me a new prompt.
Assistant: Sure, here's your new prompt: 

Explain the concept of artificial intelligence.
User: Prompt: Sure, here's your new prompt: 

Explain the concept of artificial intelligence.


Assistant: Artificial intelligence (AI) is a branch of computer science that deals with the creation of intelligent machines that can perform tasks that typically require human intelligence. AI involves the development of algorithms and computer programs that can learn from data, recognize patterns, and make decisions based on that information. 

There are two main types of AI: narrow or weak AI and general or strong AI. Narrow AI is designed to perform a specific task, such as playing chess or driving a car. It is programmed to follow a set of rules and make decisions based on those rules. General AI, on the other hand, is designed to be more flexible and adaptable. It can learn and reason like a human, and can perform a wide range of tasks across different domains.

AI is used in a variety of applications, from virtual assistants like Siri and Alexa to self-driving cars and medical diagnosis. It has the potential to revolutionize many industries and improve our lives in countless ways. However, there are also concerns about the ethical implications of AI, such as the potential for job displacement and the risk of bias in decision-making algorithms. As AI continues to evolve and become more advanced, it will be important to carefully consider these issues and ensure that the technology is used in a responsible and beneficial way.
User: Give me a new prompt.
Assistant: Sure, here's your new prompt:

Explain the greenhouse effect and its impact on climate change.
```

License
-------

This project is released under the MIT License.
