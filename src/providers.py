import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import litellm

# Automatically load .env if present
load_dotenv()

# Force local model cost map before litellm import to prevent network requests at startup
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm

# Optimization and quiet mode for litellm
litellm.telemetry = False
litellm.suppress_debug_info = True
logger = logging.getLogger("AgentProvider")


class LLMToolCallError(RuntimeError):
    """Raised when the LLM fails to output a valid tool call after max retries."""
    pass


class AgentProvider:
    """Wrapper around LLM completion API supporting multi-provider switching (Gemini, OpenAI, Anthropic)."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.2,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.5,
    ):
        raw_model = model or os.getenv("COLLUSION_MODEL", "gemini/gemini-1.5-flash")
        self.model = self._normalize_model_name(raw_model)
        self.api_key = self._resolve_api_key(api_key, self.model)
        self.api_base = api_base or os.getenv("LLM_API_BASE")
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        logger.info(f"Initialized AgentProvider for model: '{self.model}'")

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """Ensures appropriate provider prefixes for litellm (e.g. 'gemini/' prefix)."""
        model_name = model_name.strip()
        if "gemini" in model_name.lower() and not model_name.startswith("gemini/"):
            return f"gemini/{model_name}"
        return model_name

    @staticmethod
    def _resolve_api_key(api_key: Optional[str], model: str) -> Optional[str]:
        """Resolves the appropriate API key from explicit param or provider-specific env vars."""
        if api_key:
            return api_key

        model_lower = model.lower()

        # Google Gemini / Vertex
        if "gemini" in model_lower:
            return (
                os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("LLM_API_KEY")
            )

        # OpenAI
        if "gpt" in model_lower or "openai" in model_lower:
            return os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")

        # Anthropic Claude
        if "claude" in model_lower or "anthropic" in model_lower:
            return os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY")

        # Groq
        if "groq" in model_lower:
            return os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")

        # Mistral
        if "mistral" in model_lower:
            return os.getenv("MISTRAL_API_KEY") or os.getenv("LLM_API_KEY")

        # Generic fallback
        return os.getenv("LLM_API_KEY")

    def generate_action(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Any = "required",
    ) -> Dict[str, Any]:
        """Queries the model and returns a validated tool call."""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.api_base:
                    kwargs["api_base"] = self.api_base

                response = litellm.completion(**kwargs)

                choice = response.choices[0]
                message = choice.message

                tool_calls = getattr(message, "tool_calls", None)
                if not tool_calls:
                    raise ValueError(
                        f"Model did not return any tool_calls. Content: {getattr(message, 'content', '')}"
                    )

                first_call = tool_calls[0]
                func_name = first_call.function.name
                raw_args = first_call.function.arguments

                # Validate JSON arguments
                if isinstance(raw_args, str):
                    parsed_args = json.loads(raw_args)
                elif isinstance(raw_args, dict):
                    parsed_args = raw_args
                else:
                    raise ValueError(f"Unexpected arguments format: {type(raw_args)}")

                # Convert message to dictionary for serializable history / JSONL logging
                raw_message_dict = {
                    "role": "assistant",
                    "content": getattr(message, "content", None),
                    "tool_calls": [
                        {
                            "id": getattr(first_call, "id", f"call_{int(time.time()*1000)}"),
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": raw_args if isinstance(raw_args, str) else json.dumps(raw_args),
                            },
                        }
                    ],
                }

                return {
                    "name": func_name,
                    "arguments": parsed_args,
                    "tool_call_id": raw_message_dict["tool_calls"][0]["id"],
                    "raw_tool_call": first_call,
                    "raw_message": raw_message_dict,
                    "content": getattr(message, "content", None),
                }

            except Exception as e:
                last_error = e
                logger.warning(
                    f"[Attempt {attempt}/{self.max_retries}] LLM call failed for model {self.model}: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds * attempt)

        # Fail hard to prevent simulation corruption
        raise LLMToolCallError(
            f"Failed to obtain a valid tool call from {self.model} after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )
