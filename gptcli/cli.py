from prompt_toolkit import PromptSession
from openai import OpenAIError, InvalidRequestError
from rich.console import Console
from rich.markdown import Markdown
from typing import Any, Dict, Tuple
from gptcli.session import (
    ChatListener,
    InvalidArgumentError,
    ResponseStreamer,
    UserInputProvider,
)

from gptcli.term_utils import (
    StreamingMarkdownPrinter,
    parse_args,
    prompt,
)


TERMINAL_WELCOME = """
Hi! I'm here to help. Type `q` or Ctrl-D to exit, `c` or Ctrl-C to clear
the conversation, `r` or Ctrl-R to re-generate the last response.
To enter multi-line mode, enter a backslash `\\` followed by a new line.
Exit the multi-line mode by pressing ESC and then Enter (Meta+Enter).
"""


class CLIResponseStreamer(ResponseStreamer):
    def __init__(self, console: Console, markdown: bool):
        self.console = console
        self.markdown = markdown
        self.printer = StreamingMarkdownPrinter(self.console, self.markdown)
        self.first_token = True

    def __enter__(self):
        self.printer.__enter__()
        return self

    def on_next_token(self, token: str):
        if self.first_token and token.startswith(" "):
            token = token[1:]
        self.first_token = False
        self.printer.print(token)

    def __exit__(self, *args):
        self.printer.__exit__(*args)


class CLIChatListener(ChatListener):
    def __init__(self, markdown: bool):
        self.markdown = markdown
        self.console = Console()

    def on_chat_start(self):
        console = Console(width=80)
        console.print(Markdown(TERMINAL_WELCOME))

    def on_chat_clear(self):
        self.console.print("[bold]Cleared the conversation.[/bold]")

    def on_chat_rerun(self, success: bool):
        if success:
            self.console.print("[bold]Re-running the last message.[/bold]")
        else:
            self.console.print("[bold]Nothing to re-run.[/bold]")

    def on_error(self, e: Exception):
        if isinstance(e, InvalidRequestError):
            self.console.print(
                f"[red]Request Error. The last prompt was not saved: {type(e)}: {e}[/red]"
            )
        elif isinstance(e, OpenAIError):
            self.console.print(
                f"[red]API Error. Type `r` or Ctrl-R to try again: {type(e)}: {e}[/red]"
            )
        elif isinstance(e, InvalidArgumentError):
            self.console.print(f"[red]{e.message}[/red]")
        else:
            self.console.print(f"[red]Error: {type(e)}: {e}[/red]")

    def response_streamer(self) -> ResponseStreamer:
        return CLIResponseStreamer(self.console, self.markdown)


class CLIUserInputProvider(UserInputProvider):
    def __init__(self) -> None:
        self.prompt_session = PromptSession[str]()

    def get_user_input(self) -> Tuple[str, Dict[str, Any]]:
        while (next_user_input := self._request_input()) == "":
            pass

        user_input, args = self._parse_input(next_user_input)
        return user_input, args

    def _request_input(self):
        line = prompt(self.prompt_session)

        if line != "\\":
            return line

        return prompt(self.prompt_session, multiline=True)

    def _parse_input(self, input: str) -> Tuple[str, Dict[str, Any]]:
        input, args = parse_args(input)
        return input, args
