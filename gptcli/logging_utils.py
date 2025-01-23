import logging
from gptcli.completion import Message
from gptcli.session import ChatListener


class LoggingChatListener(ChatListener):
    def __init__(self):
        self.logger = logging.getLogger("gptcli-session")

    def on_chat_start(self):
        self.logger.info("Chat started")

    def on_chat_clear(self):
        self.logger.info("Cleared the conversation.")

    def on_chat_rerun(self, success: bool):
        if success:
            self.logger.info("Re-generating the last message.")

    def on_chat_back(self, x: int):
        self.logger.info(f"Going back to message pair {x}.")

    def on_error(self, e: Exception):
        self.logger.exception(e)

    def on_chat_message(self, message: Message):
        self.logger.info(f"{message['role']}: {message['content']}")
