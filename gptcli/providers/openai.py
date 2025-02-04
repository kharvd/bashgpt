import re
from typing import Iterator, List, Optional, cast
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from gptcli.completion import (
    CompletionEvent,
    CompletionProvider,
    Message,
    CompletionError,
    BadRequestError,
    MessageDeltaEvent,
    Pricing,
    UsageEvent,
)


class OpenAICompletionProvider(CompletionProvider):
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.client = OpenAI(
            api_key=api_key or openai.api_key, base_url=base_url or openai.base_url
        )

    def complete(
        self, messages: List[Message], args: dict, stream: bool = False
    ) -> Iterator[CompletionEvent]:
        kwargs = {}
        if "temperature" in args:
            kwargs["temperature"] = args["temperature"]
        if "top_p" in args:
            kwargs["top_p"] = args["top_p"]

        model = args["model"]
        if model.startswith("oai-compat:"):
            model = model[len("oai-compat:") :]

        if model.startswith("oai-azure:"):
            model = model[len("oai-azure:") :]

        try:
            if stream:
                response_iter = self.client.chat.completions.create(
                    messages=cast(List[ChatCompletionMessageParam], messages),
                    stream=True,
                    model=model,
                    stream_options={"include_usage": True},
                    **kwargs,
                )

                for response in response_iter:
                    if (
                        len(response.choices) > 0
                        and response.choices[0].finish_reason is None
                        and response.choices[0].delta.content
                    ):
                        yield MessageDeltaEvent(response.choices[0].delta.content)

                    if response.usage and (pricing := gpt_pricing(args["model"])):
                        yield UsageEvent.with_pricing(
                            prompt_tokens=response.usage.prompt_tokens,
                            completion_tokens=response.usage.completion_tokens,
                            total_tokens=response.usage.total_tokens,
                            pricing=pricing,
                        )
            else:
                response = self.client.chat.completions.create(
                    messages=cast(List[ChatCompletionMessageParam], messages),
                    model=model,
                    stream=False,
                    **kwargs,
                )
                next_choice = response.choices[0]
                if next_choice.message.content:
                    yield MessageDeltaEvent(next_choice.message.content)
                if response.usage and (pricing := gpt_pricing(args["model"])):
                    yield UsageEvent.with_pricing(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                        pricing=pricing,
                    )

        except openai.BadRequestError as e:
            raise BadRequestError(e.message) from e
        except openai.APIError as e:
            raise CompletionError(e.message) from e


GPT_3_5_TURBO_0125_PRICE_PER_TOKEN: Pricing = {
    "prompt": 0.50 / 1_000_000,
    "response": 1.50 / 1_000_000,
}

GPT_3_5_TURBO_INSTRUCT_PRICE_PER_TOKEN: Pricing = {
    "prompt": 1.50 / 1_000_000,
    "response": 2.00 / 1_000_000,
}

GPT_3_5_TURBO_1106_PRICE_PER_TOKEN: Pricing = {
    "prompt": 1.00 / 1_000_000,
    "response": 2.00 / 1_000_000,
}

GPT_3_5_TURBO_0613_PRICE_PER_TOKEN: Pricing = {
    "prompt": 1.50 / 1_000_000,
    "response": 2.00 / 1_000_000,
}

GPT_3_5_TURBO_16K_0613_PRICE_PER_TOKEN: Pricing = {
    "prompt": 3.00 / 1_000_000,
    "response": 4.00 / 1_000_000,
}

GPT_3_5_TURBO_0301_PRICE_PER_TOKEN: Pricing = {
    "prompt": 1.50 / 1_000_000,
    "response": 2.00 / 1_000_000,
}

GPT_4_PRICE_PER_TOKEN: Pricing = {
    "prompt": 30.0 / 1_000_000,
    "response": 60.0 / 1_000_000,
}

GPT_4_32K_PRICE_PER_TOKEN: Pricing = {
    "prompt": 60.0 / 1_000_000,
    "response": 120.0 / 1_000_000,
}

GPT_4_0125_PREVIEW_PRICE_PER_TOKEN: Pricing = {
    "prompt": 10.0 / 1_000_000,
    "response": 30.0 / 1_000_000,
}

GPT_4_1106_PREVIEW_PRICE_PER_TOKEN: Pricing = {
    "prompt": 10.0 / 1_000_000,
    "response": 30.0 / 1_000_000,
}

GPT_4_VISION_PREVIEW_PRICE_PER_TOKEN: Pricing = {
    "prompt": 10.0 / 1_000_000,
    "response": 30.0 / 1_000_000,
}

GPT_4_TURBO_PRICE_PER_TOKEN: Pricing = {
    "prompt": 10.0 / 1_000_000,
    "response": 30.0 / 1_000_000,
}

GPT_4_TURBO_2024_04_09_PRICE_PER_TOKEN: Pricing = {
    "prompt": 10.0 / 1_000_000,
    "response": 30.0 / 1_000_000,
}

GPT_4_O_2024_05_13_PRICE_PER_TOKEN: Pricing = {
    "prompt": 5.0 / 1_000_000,
    "response": 15.0 / 1_000_000,
}

GPT_4_O_2024_08_06_PRICE_PER_TOKEN: Pricing = {
    "prompt": 2.50 / 1_000_000,
    "response": 10.0 / 1_000_000,
}

GPT_4_O_2024_11_20_PRICE_PER_TOKEN: Pricing = {
    "prompt": 2.50 / 1_000_000,
    "response": 10.0 / 1_000_000,
}

GPT_4_O_PRICE_PER_TOKEN: Pricing = {
    "prompt": 2.50 / 1_000_000,
    "response": 10.0 / 1_000_000,
}

GPT_4_O_AUDIO_PREVIEW_PRICE_PER_TOKEN: Pricing = {
    "prompt": 2.50 / 1_000_000,
    "response": 10.0 / 1_000_000,
}
GPT_4_O_AUDIO_PREVIEW_2024_10_01_PRICE_PER_TOKEN: Pricing = {
    "prompt": 2.50 / 1_000_000,
    "response": 10.0 / 1_000_000,
}

GPT_4_O_AUDIO_PREVIEW_2024_12_17_PRICE_PER_TOKEN: Pricing = {
    "prompt": 2.50 / 1_000_000,
    "response": 10.0 / 1_000_000,
}

GPT_4_O_MINI_PRICE_PER_TOKEN: Pricing = {
    "prompt": 0.150 / 1_000_000,
    "response": 0.600 / 1_000_000,
}

GPT_4_O_MINI_2024_07_18_PRICE_PER_TOKEN: Pricing = {
    "prompt": 0.150 / 1_000_000,
    "response": 0.600 / 1_000_000,
}
GPT_4_O_MINI_AUDIO_PREVIEW_PRICE_PER_TOKEN: Pricing = {
    "prompt": 0.150 / 1_000_000,
    "response": 0.600 / 1_000_000,
}
GPT_4_O_MINI_AUDIO_PREVIEW_2024_12_17_PRICE_PER_TOKEN: Pricing = {
    "prompt": 0.150 / 1_000_000,
    "response": 0.600 / 1_000_000,
}

O_1_PRICE_PER_TOKEN: Pricing = {
    "prompt": 15.0 / 1_000_000,
    "response": 60.0 / 1_000_000,
}

O_1_2024_12_17_PRICE_PER_TOKEN: Pricing = {
    "prompt": 15.0 / 1_000_000,
    "response": 60.0 / 1_000_000,
}

O_1_PREVIEW_2024_09_12_PRICE_PER_TOKEN: Pricing = {
    "prompt": 15.0 / 1_000_000,
    "response": 60.0 / 1_000_000,
}

O_3_MINI_PRICE_PER_TOKEN: Pricing = {
    "prompt": 1.10 / 1_000_000,
    "response": 4.40 / 1_000_000,
}

O_3_MINI_2025_01_31_PRICE_PER_TOKEN: Pricing = {
    "prompt": 1.10 / 1_000_000,
    "response": 4.40 / 1_000_000,
}

CHATGPT_4O_LATEST_PRICE_PER_TOKEN: Pricing = {
    "prompt": 5.00 / 1_000_000,
    "response": 15.00 / 1_000_000,
}

DAVINCI_002_PRICE_PER_TOKEN: Pricing = {
    "prompt": 2.00 / 1_000_000,
    "response": 2.00 / 1_000_000,
}

BABBAGE_002_PRICE_PER_TOKEN: Pricing = {
    "prompt": 0.40 / 1_000_000,
    "response": 0.40 / 1_000_000,
}

PRICING_MAP: dict[str, Pricing] = {
    "davinci-002": DAVINCI_002_PRICE_PER_TOKEN,
    "babbage-002": BABBAGE_002_PRICE_PER_TOKEN,
    "gpt-3.5-turbo-0125": GPT_3_5_TURBO_0125_PRICE_PER_TOKEN,
    "gpt-3.5-turbo-instruct": GPT_3_5_TURBO_INSTRUCT_PRICE_PER_TOKEN,
    "gpt-3.5-turbo-1106": GPT_3_5_TURBO_1106_PRICE_PER_TOKEN,
    "gpt-3.5-turbo-0613": GPT_3_5_TURBO_0613_PRICE_PER_TOKEN,
    "gpt-3.5-turbo-16k-0613": GPT_3_5_TURBO_16K_0613_PRICE_PER_TOKEN,
    "gpt-3.5-turbo-0301": GPT_3_5_TURBO_0301_PRICE_PER_TOKEN,
    "gpt-4": GPT_4_PRICE_PER_TOKEN,
    "gpt-4-32k": GPT_4_32K_PRICE_PER_TOKEN,
    "gpt-4-0125-preview": GPT_4_0125_PREVIEW_PRICE_PER_TOKEN,
    "gpt-4-1106-preview": GPT_4_1106_PREVIEW_PRICE_PER_TOKEN,
    "gpt-4-vision-preview": GPT_4_VISION_PREVIEW_PRICE_PER_TOKEN,
    "gpt-4-turbo": GPT_4_TURBO_PRICE_PER_TOKEN,
    "gpt-4-turbo-2024-04-09": GPT_4_TURBO_2024_04_09_PRICE_PER_TOKEN,
    "gpt-4o-2024-05-13": GPT_4_O_2024_05_13_PRICE_PER_TOKEN,
    "gpt-4o-2024-08-06": GPT_4_O_2024_08_06_PRICE_PER_TOKEN,
    "gpt-4o-2024-11-20": GPT_4_O_2024_11_20_PRICE_PER_TOKEN,
    "gpt-4o": GPT_4_O_PRICE_PER_TOKEN,
    "gpt-4o-audio-preview": GPT_4_O_AUDIO_PREVIEW_PRICE_PER_TOKEN,
    "gpt-4o-audio-preview-2024-10-01": GPT_4_O_AUDIO_PREVIEW_2024_10_01_PRICE_PER_TOKEN,
    "gpt-4o-audio-preview-2024-12-17": GPT_4_O_AUDIO_PREVIEW_2024_12_17_PRICE_PER_TOKEN,
    "gpt-4o-mini": GPT_4_O_MINI_PRICE_PER_TOKEN,
    "gpt-4o-mini-2024-07-18": GPT_4_O_MINI_2024_07_18_PRICE_PER_TOKEN,
    "gpt-4o-mini-audio-preview": GPT_4_O_MINI_AUDIO_PREVIEW_PRICE_PER_TOKEN,
    "gpt-4o-mini-audio-preview-2024-12-17": GPT_4_O_MINI_AUDIO_PREVIEW_2024_12_17_PRICE_PER_TOKEN,
    "o1": O_1_PRICE_PER_TOKEN,
    "o1-2024-12-17": O_1_2024_12_17_PRICE_PER_TOKEN,
    "o1-preview-2024-09-12": O_1_PREVIEW_2024_09_12_PRICE_PER_TOKEN,
    "o3-mini": O_3_MINI_PRICE_PER_TOKEN,
    "o3-mini-2025-01-31": O_3_MINI_2025_01_31_PRICE_PER_TOKEN,
    "chatgpt-4o-latest": CHATGPT_4O_LATEST_PRICE_PER_TOKEN,
}


def gpt_pricing(model: str) -> Optional[Pricing]:
    """
    Returns the pricing structure for the given model name using a dictionary lookup.
    """
    return PRICING_MAP.get(model)
