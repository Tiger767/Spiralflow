from message import Message


class ChatHistory:
    def __init__(self):
        self.messages = []

    def add_message(self, message: Message):
        if not message.defined():
            raise ValueError("Message must have all defined variables.")
        self.messages.append(message.get_const())
