import os
import sys
from attr import dataclass
import platform
from typing import Any, Dict, Iterator, Optional, TypedDict, List

from gptcli.completion import (
    CompletionEvent,
    CompletionProvider,
    Message,
)
from gptcli.providers.google import GoogleCompletionProvider
from gptcli.providers.llama import LLaMACompletionProvider
from gptcli.providers.openai import OpenAICompletionProvider
from gptcli.providers.anthropic import AnthropicCompletionProvider
from gptcli.providers.cohere import CohereCompletionProvider
from gptcli.providers.azure_openai import AzureOpenAICompletionProvider
from gptcli.config import ModelConfig, AssistantConfig


CONFIG_DEFAULTS = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "top_p": 1.0,
}

DEFAULT_ASSISTANTS: Dict[str, AssistantConfig] = {
    "dev": {
        "messages": [
            {
                "role": "system",
                "content": f"You are a helpful assistant who is an expert in software development. \
You are helping a user who is a software developer. Your responses are short and concise. \
You include code snippets when appropriate. Code snippets are formatted using Markdown \
with a correct language tag. User's `uname`: {platform.uname()}",
            },
            {
                "role": "user",
                "content": "Your responses must be short and concise. Do not include explanations unless asked.",
            },
            {
                "role": "assistant",
                "content": "Understood.",
            },
        ],
    },
    "general": {
        "messages": [],
    },
    "bash": {
        "messages": [
            {
                "role": "system",
                "content": f"You output only valid and correct shell commands according to the user's prompt. \
You don't provide any explanations or any other text that is not valid shell commands. \
User's `uname`: {platform.uname()}. User's `$SHELL`: {os.environ.get('SHELL')}.",
            }
        ],
    },
}


def get_completion_provider(
    model: str,
    openai_base_url_override: Optional[str] = None,
    openai_api_key_override: Optional[str] = None,
) -> CompletionProvider:
    if model.startswith("oai-azure:"):
        return AzureOpenAICompletionProvider()
    elif model.startswith("claude"):
        return AnthropicCompletionProvider()
    elif model.startswith("llama"):
        return LLaMACompletionProvider()
    elif model.startswith("command") or model.startswith("c4ai"):
        return CohereCompletionProvider()
    elif model.startswith("gemini"):
        return GoogleCompletionProvider()
    else:
        return OpenAICompletionProvider(
            openai_base_url_override, openai_api_key_override
        )


class Assistant:
    def __init__(self, config: AssistantConfig, model_configs: Optional[Dict[str, ModelConfig]] = None):
        self.config = config
        self.model_configs = model_configs or {}

    @classmethod
    def from_config(cls, name: str, config: AssistantConfig, model_configs: Optional[Dict[str, ModelConfig]] = None):
        config = config.copy()
        if name in DEFAULT_ASSISTANTS:
            # Merge the config with the default config
            # If a key is in both, use the value from the config
            default_config = DEFAULT_ASSISTANTS[name]
            for key in [*config.keys(), *default_config.keys()]:
                if config.get(key) is None:
                    config[key] = default_config[key]

        return cls(config, model_configs=model_configs)

    def init_messages(self) -> List[Message]:
        return self.config.get("messages", [])[:]

    def _param(self, param: str) -> Any:
        # Use the value from the config if exists
        # Otherwise, use the default value
        return self.config.get(param, CONFIG_DEFAULTS.get(param, None))

    def complete_chat(self, messages, stream: bool = True) -> Iterator[CompletionEvent]:
        model = self._param("model")
        # Check if there is a model configuration override for this model.
        if model in self.model_configs:
            model_conf = self.model_configs[model]
            print(model_conf)
            openai_api_key_override = model_conf['api_key'] or self.config.get("openai_api_key_override")
            openai_base_url_override = model_conf['base_url'] or self.config.get("openai_base_url_override")
            pricing_override = model_conf['pricing']
        else:
            openai_api_key_override = self._param("openai_api_key_override")
            openai_base_url_override = self._param("openai_base_url_override")
            pricing_override = None

        completion_provider = get_completion_provider(
            model,
            openai_base_url_override,
            openai_api_key_override,
        )
        return completion_provider.complete(
            messages,
            {
                "model": model,
                "temperature": float(self._param("temperature")),
                "top_p": float(self._param("top_p")),
                # Pass along the pricing override if available.
                "pricing": pricing_override,
            },
            stream,
        )


@dataclass
class AssistantGlobalArgs:
    assistant_name: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


def init_assistant(
    args: AssistantGlobalArgs,
    custom_assistants: Dict[str, AssistantConfig],
    model_configs: Optional[Dict[str, ModelConfig]] = None,
) -> Assistant:
    name = args.assistant_name
    if name in custom_assistants:
        assistant = Assistant.from_config(name, custom_assistants[name], model_configs=model_configs)
    elif name in DEFAULT_ASSISTANTS:
        assistant = Assistant.from_config(name, DEFAULT_ASSISTANTS[name], model_configs=model_configs)
    else:
        print(f"Unknown assistant: {name}")
        sys.exit(1)

    # Override config with command line arguments
    if args.temperature is not None:
        assistant.config["temperature"] = args.temperature
    if args.model is not None:
        assistant.config["model"] = args.model
    if args.top_p is not None:
        assistant.config["top_p"] = args.top_p
    return assistant
