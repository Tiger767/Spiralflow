from typing import Dict, List, Tuple, Union
import openai

from .message import Message


class ChatLLM:
    """
    A class for chat completion using the GPT model.
    """

    def __init__(
        self, gpt_model: str = "gpt-3.5-turbo", stream_hook=None, **kwargs
    ) -> None:
        """
        Initializes the ChatLLM class with the given parameters.

        :param gpt_model: GPT model to use for chat completion.
        :param stream_hook: Enables streaming to this function.
        """
        self.gpt_model = gpt_model
        self.model_params = kwargs
        self.stream_hook = stream_hook

    def __call__(self, messages: List[Message]) -> Tuple[str, str, Union[Dict, List[Dict]]]:
        """
        Generates a response using the GPT model based on the input messages.

        :param messages: List of messages to use for chat completion.
        :return: Response from the chat completion with content, role, and metadata.
        """
        if self.stream_hook is not None:
            completion = openai.chat.completions.create(
                model=self.gpt_model, messages=messages, stream=True, **self.model_params
            )
            role = None
            responses = []
            full_message = []
            for chunk in completion:
                responses.append(chunk)
                if chunk.choices[0].delta.role is not None:
                    role = chunk.choices[0].delta.role
                content = chunk.choices[0].delta.content
                self.stream_hook(content, role, chunk)
                if content is not None:
                    full_message.append(content)
            message = "".join(full_message)
            return message, role, responses
        else:
            response = openai.chat.completions.create(
                model=self.gpt_model, messages=messages, **self.model_params
            )
            message = response.choices[0].message

            return message.content, message.role, response
