"""
OpenRouter API client — sync and async streaming with reasoning detection.

This module communicates with OpenRouter, which is an API gateway that
provides access to many LLMs (DeepSeek, Llama, GPT, Claude, etc.) through
a single OpenAI-compatible API. We use the openai Python library.

OpenRouter's free tier has rate limits (check https://openrouter.ai/docs/limits
for current limits) but is good for testing. Paid tier removes these limits.

Two interfaces provided:
1. Synchronous call() for simple request/response (summarization).
2. Async stream_with_reasoning() for streaming (intervention).

We maintain TWO client instances because:
- sync client: used for summarization (runs via asyncio.to_thread)
- async client: used for streaming (needs async for await on chunks)
"""

import time
import logging
from openai import AsyncOpenAI, OpenAI, APIError, RateLimitError

logger = logging.getLogger("palbot")


class OpenRouterClient:
    """
    Wraps the OpenAI SDK configured for OpenRouter's API endpoint.
    
    OpenAI SDK is a Python library that sends HTTP requests to any
    OpenAI-compatible API. By setting base_url to OpenRouter's endpoint,
    all calls go to OpenRouter instead of OpenAI.
    
    We keep two client instances:
      - self.client (sync OpenAI): for one-shot requests via asyncio.to_thread
      - self.async_client (AsyncOpenAI): for streaming with await
    
    Args:
        config: Config object with API key, model selection, and settings.
    """
    
    def __init__(self, config):
        # === SYNC CLIENT ===
        # OpenAI(api_key=..., base_url=...) creates a synchronous client.
        # All API calls block the current thread until complete.
        # We run this via asyncio.to_thread() so it doesn't block the
        # asyncio event loop.
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY,
            default_headers={
                # These two headers are REQUIRED by OpenRouter's free tier.
                # Without them, free models return HTTP 402 Payment Required.
                "HTTP-Referer": "https://github.com/taya/MAG-pal",
                "X-Title": "MAG-pal",
            },
            # Retry up to 3 times on transient failures (rate limits, timeouts)
            max_retries=3,
            # Total timeout for the entire request (connect + response)
            timeout=60.0,
        )

        # === ASYNC CLIENT ===
        # AsyncOpenAI() is identical to OpenAI() but all methods return
        # coroutines (things you await) instead of blocking.
        # This is used for streaming — we get chunks one at a time
        # while the event loop handles other tasks.
        self.async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": "https://github.com/taya/MAG-pal",
                "X-Title": "MAG-pal",
            },
            max_retries=3,
            timeout=60.0,
        )

        self.model = config.MODEL
        self.max_tokens = config.MAX_TOKENS
        self.temperature = config.TEMPERATURE

        # Reasoning tag detection config.
        # Some models wrap internal reasoning in XML-style tags.
        # DeepSeek-R1 uses <think> and </think>.
        # You can override these in .env if needed.
        self.reasoning_tag_open = getattr(config, 'REASONING_TAG_OPEN', '<think>')
        self.reasoning_tag_close = getattr(config, 'REASONING_TAG_CLOSE', '</think>')

    def close(self):
        """Clean up the HTTP client's connection pool on shutdown."""
        self.client.close()

    # ===== SYNCHRONOUS CALL =====
    # Used for summarization. Runs in a thread pool via asyncio.to_thread.

    def call(self, messages: list) -> tuple[str, dict]:
        """
        Send messages to OpenRouter synchronously.
        
        This is a SYNCHRONOUS method. It blocks the current thread until
        the API responds. In bot.py, it's called via asyncio.to_thread()
        so it runs in a separate thread and doesn't block the event loop.
        
        Args:
            messages: List of message dicts, each with "role" and "content":
                      [{"role": "system", "content": "You are..."},
                       {"role": "user", "content": "Hello!"}]
        
        Returns:
            Tuple of (response_text, usage_dict) where usage_dict contains:
            - prompt_tokens: tokens in the input messages
            - completion_tokens: tokens in the AI's response
            - latency: seconds elapsed
        
        Exceptions are caught and re-raised with logging so the caller
        (bot.py) knows something went wrong.
        """
        start = time.time()
        try:
            # The actual API call. This is exactly like OpenAI's format.
            # response is a ChatCompletion object with:
            #   response.choices[0].message.content (the text)
            #   response.usage (tokens used)
            #   response.model (which model responded)
            #
            # model=self.model selects which LLM to use
            # messages=messages is the conversation history
            # max_tokens limits response length
            # temperature controls creativity (0.0 = deterministic, 1.0 = chaotic)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            latency = time.time() - start
            usage = response.usage
            
            # response.choices[0].message.content extracts the AI's text.
            # .choices is a list — the API can return multiple candidates,
            # but we always use the first one (index 0).
            # .message.content is the actual response string.
            # or "" handles the edge case where the response is None.
            return (response.choices[0].message.content or ""), {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "latency": round(latency, 2),
            }
            
        # RateLimitError: HTTP 429 — we hit OpenRouter's rate limit.
        # The SDK handles retries with exponential backoff (max_retries=3),
        # but if all retries are exhausted, this exception is raised.
        except RateLimitError as e:
            logger.warning(f"OpenRouter rate limited: {e}")
            raise
            
        # APIError: HTTP 400, 500, etc. — something went wrong on the server.
        except APIError as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
            
        # Catch-all: network errors, DNS failures, timeouts, etc.
        except Exception as e:
            logger.error(f"OpenRouter unexpected error: {e}")
            raise

    # ===== ASYNC STREAMING WITH REASONING DETECTION =====
    # This is used by the streaming intervention system.
    # It's an ASYNC GENERATOR that yields (token, phase) tuples.

    async def stream_with_reasoning(self, messages: list):
        """
        Stream tokens from OpenRouter with reasoning-phase detection.
        
        This is an ASYNC GENERATOR — meaning it uses 'yield' instead of
        'return'. Each yield gives one token to the caller, and the
        generator pauses until the caller asks for the next one.
        
        The caller (bot.py:_stream_with_intervention) does:
          async for token, phase in self.openrouter.stream_with_reasoning(messages):
        
        Each iteration gets one (token, phase) tuple from this yield.
        
        Two methods for detecting reasoning:
        
        Method 1: reasoning_content field
          Some API models emit a separate delta.reasoning_content field.
          When this field appears, we yield with phase='reasoning'.
          When it disappears, we yield with phase='output'.
          
        Method 2: Tag-based (<think>/</think>)
          Configured via REASONING_TAG_OPEN/CLOSE in .env.
          Content between tags is phase='reasoning', rest is 'output'.
          Tags are stripped from the content before yielding.
        
        Args:
            messages: Same format as call() — list of {"role", "content"} dicts.
        
        Yields:
            (token_text: str, phase: 'reasoning' | 'output')
        """
        
        # First: create the stream. This is an async call to OpenRouter.
        # stream=True tells the API to send tokens one at a time.
        # If this fails (network, auth, rate limit), the error is
        # caught here and re-raised with logging.
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )
        except RateLimitError as e:
            logger.warning(f"OpenRouter rate limited (stream): {e}")
            raise
        except APIError as e:
            logger.error(f"OpenRouter API error (stream): {e}")
            raise
        except Exception as e:
            logger.error(f"OpenRouter unexpected error (stream): {e}")
            raise

        # State machine flags for reasoning detection:
        in_reasoning = False     # currently inside a reasoning block
        saw_reasoning = False    # latch — reasoning ever detected in this stream?

        # Second: iterate over the stream chunks.
        # Each chunk is a ChatCompletionChunk object with:
        #   chunk.choices[0].delta (the token data)
        try:
            async for chunk in stream:
                # Guard: some chunks have empty choices (keep-alive signals)
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                if not delta:
                    continue

                # --- Method 1: reasoning_content field ---
                # Some models (DeepSeek via API) emit a separate field
                # for reasoning tokens vs output tokens.
                # reasoning_content appears ONLY during reasoning phase.
                reasoning = getattr(delta, 'reasoning_content', None)
                if reasoning:
                    in_reasoning = True
                    saw_reasoning = True       # latch on permanently
                    yield reasoning, 'reasoning'
                    continue

                content = delta.content
                if not content:
                    continue

                # --- Reasoning→output transition for Method 1 models ---
                # When saw_reasoning is True AND we just got content
                # WITHOUT reasoning_content, the model switched to output.
                # The "real reasoning" is done.
                if saw_reasoning and in_reasoning:
                    in_reasoning = False

                # --- Method 2: Tag-based reasoning detection ---
                # For models that use <think> tags in the content itself.
                tag_open = self.reasoning_tag_open
                tag_close = self.reasoning_tag_close

                if tag_open and tag_open in content:
                    in_reasoning = True
                    content = content.replace(tag_open, '')
                if tag_close and tag_close in content:
                    in_reasoning = False
                    content = content.replace(tag_close, '')

                if content:
                    yield content, 'reasoning' if in_reasoning else 'output'
                    
        # Error handling for the streaming loop itself
        except (APIError, RateLimitError) as e:
            logger.error(f"OpenRouter stream error: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenRouter stream unexpected error: {e}")
            raise
