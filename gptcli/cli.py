from typing import Optional

from openai import BadRequestError, OpenAIError
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.key_binding.bindings import named_commands
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from gptcli.session import (
    ALL_COMMANDS,
    COMMAND_CLEAR,
    COMMAND_QUIT,
    COMMAND_RERUN,
    ChatListener,
    InvalidArgumentError,
    ResponseStreamer,
    UserInputProvider,
)

TERMINAL_WELCOME = """
Hi! I'm here to help. Type `:q` or Ctrl-D to exit, `:c` or Ctrl-C and Enter to clear
the conversation, `:r` or Ctrl-R to re-generate the last response.
Type `:b X` where X is a message pair number to go back to that message pair.
To enter multi-line mode, enter a backslash `\\` followed by a new line.
Exit the multi-line mode by pressing ESC and then Enter (Meta+Enter).
Try `:?` for help.
"""


class StreamingMarkdownPrinter:
    def __init__(self, console: Console, markdown: bool):
        self.console = console
        self.current_text = ""
        self.markdown = markdown
        self.live: Optional[Live] = None

    def __enter__(self) -> "StreamingMarkdownPrinter":
        if self.markdown:
            self.live = Live(
                console=self.console, auto_refresh=False, vertical_overflow="visible"
            )
            self.live.__enter__()
        return self

    def print(self, text: str):
        self.current_text += text
        if self.markdown:
            assert self.live
            content = Markdown(self.current_text, style="green")
            self.live.update(content)
            self.live.refresh()
        else:
            self.console.print(Text(text, style="green"), end="")

    def __exit__(self, *args):
        if self.markdown:
            assert self.live
            self.live.__exit__(*args)
        self.console.print()


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

    def on_chat_back(self, x: int):
        self.console.print(f"[bold]Going back to message pair {x}.[/bold]")

    def on_error(self, e: Exception):
        if isinstance(e, BadRequestError):
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


class CLIFileHistory(FileHistory):
    def append_string(self, string: str) -> None:
        if string in ALL_COMMANDS:
            return
        return super().append_string(string)


class CLIUserInputProvider(UserInputProvider):
    def __init__(self, history_filename) -> None:
        self.prompt_session = PromptSession[str](
            history=CLIFileHistory(history_filename)
        )

    def get_user_input(self) -> str:
        while (next_user_input := self._request_input()) == "":
            pass

        return next_user_input

    def prompt(self, multiline=False):
        bindings = KeyBindings()

        bindings.add("c-a")(named_commands.get_by_name("beginning-of-line"))
        bindings.add("c-b")(named_commands.get_by_name("backward-char"))
        bindings.add("c-e")(named_commands.get_by_name("end-of-line"))
        bindings.add("c-f")(named_commands.get_by_name("forward-char"))
        bindings.add("c-left")(named_commands.get_by_name("backward-word"))
        bindings.add("c-right")(named_commands.get_by_name("forward-word"))

        @bindings.add("c-c")
        def _(event: KeyPressEvent):
            if len(event.current_buffer.text) == 0 and not multiline:
                event.current_buffer.text = COMMAND_CLEAR[0]
                event.current_buffer.cursor_right(len(COMMAND_CLEAR[0]))
            else:
                event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

        @bindings.add("c-d")
        def _(event: KeyPressEvent):
            if len(event.current_buffer.text) == 0:
                if not multiline:
                    event.current_buffer.text = COMMAND_QUIT[0]
                event.current_buffer.validate_and_handle()

        @bindings.add("c-r")
        def _(event: KeyPressEvent):
            if len(event.current_buffer.text) == 0:
                event.current_buffer.text = COMMAND_RERUN[0]
                event.current_buffer.validate_and_handle()

        try:
            return self.prompt_session.prompt(
                "> " if not multiline else "multiline> ",
                vi_mode=True,
                multiline=multiline,
                enable_open_in_editor=True,
                key_bindings=bindings,
            )
        except KeyboardInterrupt:
            return ""

    def _request_input(self):
        line = self.prompt()

        if line != "\\":
            return line

        return self.prompt(multiline=True)
