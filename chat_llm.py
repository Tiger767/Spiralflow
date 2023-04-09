from typing import List, Tuple
import openai

from message import Message


class ChatLLM:
    def __init__(self, gpt_model: str = 'gpt-3.5-turbo', stream: bool = False) -> None:
        """
        :param gpt_model: GPT model to use for chat completion.
        :param stream: Whether to use stream mode.
        """
        self.gpt_model = gpt_model
        self.stream = stream

    def __call__(self, messages: List[Message]) -> Tuple[str, str]:
        """
        :param messages: List of messages to use for chat completion.
        :return: Response from the chat completion.
        """
        if self.stream:
            raise NotImplemented('Stream mode is not implemented yet.')
        else:
            response = openai.ChatCompletion.create(
                model=self.gpt_model,
                messages=messages
            )
            response = response['choices'][0]['message']
            
            return response['content'], response['role']
