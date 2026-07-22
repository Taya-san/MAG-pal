"""
OpenRouter API client — sync and async streaming with reasoning detection.

OpenRouter is a gateway to many LLMs (DeepSeek, Llama, GPT, Claude, etc.)
through a single OpenAI-compatible API. Free tier is available with rate
limits (20 req/min, 50 req/day).

This client provides two interfaces:
1. Synchronous call() for simple request/response (summarization).
2. Async stream_with_reasoning() for streaming with reasoning-token
   detection — used by the streaming intervention system.
"""

import time
import logging
from openai import AsyncOpenAI, OpenAI, APIError, RateLimitError

logger = logging.getLogger("palbot")


class OpenRouterClient:
    """
    Wraps OpenAI SDK configured for OpenRouter's endpoint.
    
    Maintains two client instances:
    - client (sync OpenAI): for summarization and fallback non-streaming calls.
    - async_client (AsyncOpenAI): for streaming with intervention scanning.
    
    Args:
        config: Config object with OPENROUTER_API_KEY, MODEL, MAX_TOKENS,
                TEMPERATURE, and optional REASONING_TAG_OPEN/CLOSE.
    """
    
    def __init__(self, config):
        # Synchronous client for summarization and fallback
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": "https://github.com/taya/MAG-pal",
                "X-Title": "MAG-pal",
            },
            max_retries=3,
            timeout=60.0,
        )
        
        # Async client for streaming with intervention
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

        # Configurable reasoning tags for models that wrap reasoning
        # in XML-style tags (e.g., <think>...</think>). Override in .env
        # if your model uses different delimiters.
        self.reasoning_tag_open = getattr(config, 'REASONING_TAG_OPEN', '<think>')
        self.reasoning_tag_close = getattr(config, 'REASONING_TAG_CLOSE', '</think>')

    def close(self):
        """Clean up the HTTP client's connection pool on shutdown."""
        self.client.close()

    # ===== SYNCHRONOUS CALL (summarization, fallback) =====

    def call(self, messages: list) -> tuple[str, dict]:
        """
        Send messages to OpenRouter synchronously.
        
        Args:
            messages: List of {"role": ..., "content": ...} dicts.
        
        Returns:
            (response_text, usage_dict) where usage_dict has
            prompt_tokens, completion_tokens, and latency in seconds.
        
        This is a synchronous method called via asyncio.to_thread()
        in bot.py so it doesn't block the event loop.
        """
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            latency = time.time() - start
            usage = response.usage
            return (response.choices[0].message.content or ""), {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "latency": round(latency, 2),
            }
        except RateLimitError as e:
            logger.warning(f"OpenRouter rate limited: {e}")
            raise
        except APIError as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenRouter unexpected error: {e}")
            raise

    # ===== ASYNC STREAMING WITH REASONING DETECTION =====

    async def stream_with_reasoning(self, messages: list):
        """
        Stream tokens from OpenRouter with reasoning-phase detection.
        
        Returns an async generator yielding (token_text, phase) tuples
        where phase is either 'reasoning' or 'output'.
        
        Two detection methods are supported:
        
        Method 1: reasoning_content field
          Some models (DeepSeek via API) emit a separate
          delta.reasoning_content field. When this field appears,
          tokens are labeled 'reasoning'. When it disappears and
          delta.content starts flowing, tokens switch to 'output'.
        
        Method 2: Tag-based detection (<think>/</think>)
          Configurable via REASONING_TAG_OPEN/CLOSE. Tokens between
          open and close tags are labeled 'reasoning'. Tags are
          stripped from the content before yielding.
        
        Edge case handling:
          - saw_reasoning latch: tracks if reasoning ever occurred,
            used to detect the reasoning→output transition for
            Method 1 models.
          - in_reasoning flag: toggles. Used by Method 2 (tags)
            and reset by the transition detector for Method 1.
          - Guard: chunk.choices may be empty; delta may be None.
        
        Args:
            messages: List of {"role": ..., "content": ...} dicts.
        
        Yields:
            (text: str, phase: 'reasoning' | 'output')
        """
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

        in_reasoning = False      # currently inside a reasoning block
        saw_reasoning = False     # latch: reasoning ever detected in this stream

        try:
            async for chunk in stream:
                # Guard against empty/missing choices
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if not delta:
                    continue

                # --- Method 1: reasoning_content field ---
                reasoning = getattr(delta, 'reasoning_content', None)
                if reasoning:
                    in_reasoning = True
                    saw_reasoning = True
                    yield reasoning, 'reasoning'
                    continue

                content = delta.content
                if not content:
                    continue

                # --- Reasoning→output transition for Method 1 models ---
                # When saw_reasoning is True and we get content without
                # reasoning_content, the model switched to output phase.
                if saw_reasoning and in_reasoning:
                    in_reasoning = False

                # --- Method 2: Tag-based reasoning detection ---
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
        
        except (APIError, RateLimitError) as e:
            logger.error(f"OpenRouter stream error: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenRouter stream unexpected error: {e}")
            raise
