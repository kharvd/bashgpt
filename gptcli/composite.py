from gptcli.completion import Message, UsageEvent
from gptcli.session import ChatListener, ResponseStreamer


from typing import List, Optional


class CompositeResponseStreamer(ResponseStreamer):
    def __init__(self, streamers: List[ResponseStreamer]):
        self.streamers = streamers

    def __enter__(self):
        for streamer in self.streamers:
            streamer.__enter__()
        return self

    def on_next_token(self, token: str):
        for streamer in self.streamers:
            streamer.on_next_token(token)

    def __exit__(self, *args):
        for streamer in self.streamers:
            streamer.__exit__(*args)


class CompositeChatListener(ChatListener):
    def __init__(self, listeners: List[ChatListener]):
        self.listeners = listeners

    def on_chat_start(self):
        for listener in self.listeners:
            listener.on_chat_start()

    def on_chat_clear(self):
        for listener in self.listeners:
            listener.on_chat_clear()

    def on_chat_rerun(self, success: bool):
        for listener in self.listeners:
            listener.on_chat_rerun(success)

    def on_chat_back(self, x: int):
        for listener in self.listeners:
            listener.on_chat_back(x)

    def on_error(self, e: Exception):
        for listener in self.listeners:
            listener.on_error(e)

    def response_streamer(self) -> ResponseStreamer:
        return CompositeResponseStreamer(
            [listener.response_streamer() for listener in self.listeners]
        )

    def on_chat_message(self, message: Message):
        for listener in self.listeners:
            listener.on_chat_message(message)

    def on_chat_response(
        self,
        messages: List[Message],
        response: Message,
        usage: Optional[UsageEvent],
    ):
        for listener in self.listeners:
            listener.on_chat_response(messages, response, usage)
