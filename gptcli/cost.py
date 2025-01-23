from gptcli.assistant import Assistant
from gptcli.completion import Message, UsageEvent
from gptcli.session import ChatListener

from rich.console import Console

import logging
from typing import List, Optional


class PriceChatListener(ChatListener):
    def __init__(self, assistant: Assistant):
        self.assistant = assistant
        self.current_spend = 0
        self.logger = logging.getLogger("gptcli-price")
        self.console = Console()

    def on_chat_clear(self):
        self.current_spend = 0

    def on_chat_response(
        self,
        messages: List[Message],
        response: Message,
        usage: Optional[UsageEvent] = None,
    ):
        if usage is None:
            return

        model = self.assistant._param("model")
        num_tokens = usage.total_tokens
        cost = usage.cost
        message_idx = len(messages) // 2

        if cost is None:
            self.logger.error(f"Cannot get cost information for model {model}")
            return

        self.current_spend += cost
        self.logger.info(f"Token usage {num_tokens}")
        self.logger.info(f"Message price (model: {model}): ${cost:.3f}")
        self.logger.info(f"Current spend: ${self.current_spend:.3f}")
        self.console.print(
            f"Message pair #: {message_idx} | Tokens: {num_tokens} | Price: ${cost:.3f} | Total: ${self.current_spend:.3f}",
            justify="right",
            style="dim",
        )
