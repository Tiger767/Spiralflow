import openai

class ChatLLM:
    def __init__(self, gpt_model: str = 'gpt-3.5-turbo'):
        self.gpt_model = gpt_model

    def __call__(self, messages) -> str:
        response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=messages
        )
        return response['choices'][0]['message']['content']
