# openrouter.py
# Client for the OpenRouter API (gateway to many LLMs).
#
# OpenRouter gives access to DeepSeek, Llama, GPT, Claude, etc.
# through a single OpenAI-compatible API. Free tier is available
# with rate limits (20 req/min, 50 req/day).
#
# We use the SYNC OpenAI client and run it in a thread pool
# (via asyncio.to_thread in bot.py). This is simpler than
# AsyncOpenAI and avoids compatibility issues.

import time
import logging
from openai import OpenAI, APIError, RateLimitError

logger = logging.getLogger("palbot")


class OpenRouterClient:
    # Wraps the OpenAI SDK configured for OpenRouter's endpoint.

    def __init__(self, config):
        # The OpenAI SDK can point to any compatible API.
        # Setting base_url to OpenRouter makes all calls go there.

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY,

            # These two headers are REQUIRED by OpenRouter's free tier.
            # Without them, free models return HTTP 402 Payment Required.
            # HTTP-Referer: any URL identifying your app
            # X-Title: any name for your app
            default_headers={
                "HTTP-Referer": "https://github.com/taya/MAG-pal",
                "X-Title": "MAG-pal",
            },

            # Retry up to 3 times on transient failures (rate limits, timeouts)
            max_retries=3,

            # Total timeout for the entire request (connect + response)
            # Prevents the bot from hanging forever if OpenRouter is slow
            timeout=60.0,
        )

        # Store model config for use in call()
        self.model = config.MODEL
        self.max_tokens = config.MAX_TOKENS
        self.temperature = config.TEMPERATURE

    def close(self):
        # Clean up the HTTP client's connection pool.
        # Called when the bot shuts down.
        self.client.close()

    def call(self, messages: list) -> tuple[str, dict]:
        # Sends messages to OpenRouter and returns (response_text, usage_stats).
        #
        # messages format: list of dicts like:
        #   [{"role": "system", "content": "..."},
        #    {"role": "user", "content": "..."}]
        #
        # Returns:
        #   response_text: the AI's message content (string)
        #   usage: dict with prompt_tokens, completion_tokens, latency
        #
        # This is a SYNCHRONOUS method. It's called via asyncio.to_thread()
        # in bot.py so it doesn't block the event loop.

        start = time.time()

        try:
            # The actual API call — identical to OpenAI's format
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            latency = time.time() - start

            # Extract usage info if available (free models might not return it)
            usage = response.usage

            return (response.choices[0].message.content or ""), {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "latency": round(latency, 2),
            }

        except RateLimitError as e:
            # OpenRouter rate limit hit (429)
            # The OpenAI SDK handles retry with backoff, but if all
            # retries are exhausted, this exception is raised.
            logger.warning(f"OpenRouter rate limited: {e}")
            raise

        except APIError as e:
            # Other API errors (500, 400, etc.)
            logger.error(f"OpenRouter API error: {e}")
            raise

        except Exception as e:
            # Catch-all: network errors, DNS failures, timeouts, etc.
            logger.error(f"OpenRouter unexpected error: {e}")
            raise
