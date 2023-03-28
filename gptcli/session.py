from abc import abstractmethod
import logging
from gptcli.assistant import Assistant, Message, ModelOverrides
from gptcli.term_utils import COMMAND_CLEAR, COMMAND_QUIT, COMMAND_RERUN
from openai import InvalidRequestError, OpenAIError
from typing import Any, Dict, List, Tuple


class ResponseStreamer:
    def __enter__(self):
        pass

    def on_next_token(self, token: str):
        pass

    def __exit__(self, *args):
        pass


class ChatListener:
    def on_chat_start(self):
        pass

    def on_chat_clear(self):
        pass

    def on_chat_rerun(self, success: bool):
        pass

    def on_error(self, error: Exception):
        pass

    @abstractmethod
    def response_streamer(self) -> ResponseStreamer:
        pass

    def on_chat_message(self, message: Message):
        pass


class UserInputProvider:
    def get_user_input(self) -> Tuple[str, ModelOverrides]:
        pass


def InvalidArgumentError(Exception):
    def __init__(self, message: str):
        self.message = message


class ChatSession:
    def __init__(
        self,
        assistant: Assistant,
        listener: ChatListener,
        input_provider: UserInputProvider,
    ):
        self.assistant = assistant
        self.messages = assistant.init_messages()
        self.user_prompts: List[Tuple[str, ModelOverrides]] = []
        self.listener = listener
        self.input_provider = input_provider

    def _clear(self):
        self.messages = self.assistant.init_messages()
        self.user_prompts = []
        self.listener.on_chat_clear()
        logging.info("Cleared the conversation.")

    def _rerun(self):
        if len(self.user_prompts) == 0:
            self.listener.on_chat_rerun(False)
            return

        if self.messages[-1]["role"] == "assistant":
            self.messages = self.messages[:-1]

        logging.info("Re-generating the last message.")
        self.listener.on_chat_rerun(True)
        _, args = self.user_prompts[-1]
        self._respond(args)

    def _respond(self, args: ModelOverrides) -> bool:
        """
        Respond to the user's input and return whether the assistant's response was saved.
        """
        next_response = ""
        try:
            completion_iter = self.assistant.complete_chat(
                self.messages, override_params=args
            )

            with self.listener.response_streamer() as stream:
                for response in completion_iter:
                    next_response += response
                    stream.on_next_token(response)
        except KeyboardInterrupt:
            # If the user interrupts the chat completion, we'll just return what we have so far
            pass
        except InvalidRequestError as e:
            logging.exception(e)
            self.listener.on_error(e)
            return False
        except OpenAIError as e:
            logging.exception(e)
            self.listener.on_error(e)
            return True

        logging.info("Assistant: %s", next_response)
        next_response = {"role": "assistant", "content": next_response}
        self.messages.append(next_response)
        self.listener.on_chat_message(next_response)
        return True

    def _validate_args(self, args: Dict[str, Any]):
        for key in args:
            supported_overrides = self.assistant.supported_overrides()
            if key not in supported_overrides:
                self.listener.on_error(
                    InvalidArgumentError(
                        f"Invalid argument: {key}. Allowed arguments: {supported_overrides}"
                    )
                )
                return False
        return True

    def _add_user_message(self, user_input: str, args: ModelOverrides):
        logging.info("User: %s", user_input)
        user_message = {"role": "user", "content": user_input}
        self.messages.append(user_message)
        self.listener.on_chat_message(user_message)
        self.user_prompts.append((user_message, args))

    def _rollback_user_message(self):
        self.messages = self.messages[:-1]
        self.user_prompts = self.user_prompts[:-1]

    def loop(self):
        self.listener.on_chat_start()

        while True:
            user_input, args = self.input_provider.get_user_input()
            if not self._validate_args(args):
                continue

            if user_input in COMMAND_QUIT:
                break
            elif user_input in COMMAND_CLEAR:
                self._clear()
                continue
            elif user_input in COMMAND_RERUN:
                self._rerun()
                continue

            self._add_user_message(user_input, args)
            response_saved = self._respond(args)
            if not response_saved:
                self._rollback_user_message()