from typing import Dict, List, Tuple
import openai

from .message import Message


class ChatLLM:
    """
    A class for chat completion using the GPT model.
    """

    def __init__(
        self, gpt_model: str = "gpt-3.5-turbo", stream=False, **kwargs
    ) -> None:
        """
        Initializes the ChatLLM class with the given parameters.

        :param gpt_model: GPT model to use for chat completion.
        :param stream: Whether to use stream mode.
        """
        self.gpt_model = gpt_model
        self.model_params = kwargs
        self.stream = stream

    def __call__(self, messages: List[Message]) -> Tuple[str, str, Dict]:
        """
        Generates a response using the GPT model based on the input messages.

        :param messages: List of messages to use for chat completion.
        :return: Response from the chat completion with content, role, and metadata.
        """
        if self.stream:
            raise NotImplementedError("Stream mode is not implemented yet.")
        else:
            response = openai.ChatCompletion.create(
                model=self.gpt_model, messages=messages, **self.model_params
            )
            message = response["choices"][0]["message"]

            return message["content"], message["role"], response
