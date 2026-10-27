import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import openai

# Automatically load .env if present
load_dotenv()

logger = logging.getLogger("AgentProvider")


class LLMToolCallError(RuntimeError):
    """Raised when the LLM fails to output a valid tool call after max retries."""
    pass


class AgentProvider:
    """Wrapper around LLM completion API supporting multi-provider switching (Gemini, OpenAI, Anthropic/Groq)."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.2,
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
    ):
        raw_model = model or os.getenv("COLLUSION_MODEL", "gemini-3.8-flash")
        self.model, self.provider_type = self._normalize_model(raw_model)
        self.api_key = self._resolve_api_key(api_key, self.provider_type)
        self.api_base = self._resolve_base_url(api_base, self.provider_type)
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        # Configure OpenAI-compatible client
        self.client = openai.OpenAI(
            api_key=self.api_key or "dummy_key",
            base_url=self.api_base,
        )

        logger.info(
            f"Initialized AgentProvider for model: '{self.model}' (provider: {self.provider_type}, base_url: {self.api_base})"
        )

    @staticmethod
    def _normalize_model(model_name: str) -> tuple[str, str]:
        """Returns clean model name and provider type."""
        name = model_name.strip()
        name_lower = name.lower()

        if "gemini" in name_lower or "gemma" in name_lower:
            # Strip litellm style 'gemini/' or 'models/' prefix for OpenAI endpoint compatibility
            clean_name = name.split("/")[-1]
            return clean_name, "gemini"
        elif "deepseek" in name_lower:
            clean_name = name.split("/")[-1]
            return clean_name, "deepseek"
        elif "claude" in name_lower or "anthropic" in name_lower:
            return name, "anthropic"
        elif "groq" in name_lower:
            clean_name = name.split("/")[-1]
            return clean_name, "groq"
        else:
            return name, "openai"

    @staticmethod
    def _resolve_base_url(api_base: Optional[str], provider_type: str) -> Optional[str]:
        """Resolves base URL based on provider type."""
        if api_base:
            return api_base
        if os.getenv("LLM_API_BASE"):
            return os.getenv("LLM_API_BASE")

        if provider_type == "gemini":
            return "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif provider_type == "deepseek":
            return "https://api.deepseek.com"
        elif provider_type == "groq":
            return "https://api.groq.com/openai/v1"
        return None

    @staticmethod
    def _resolve_api_key(api_key: Optional[str], provider_type: str) -> Optional[str]:
        """Resolves the appropriate API key from environment."""
        if api_key:
            return api_key

        if provider_type == "gemini":
            return (
                os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("LLM_API_KEY")
            )
        elif provider_type == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
        elif provider_type == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY")
        elif provider_type == "groq":
            return os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
        else:
            return os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")

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
                    # DeepSeek reasoning/thinking models require 'auto' rather than 'required'
                    if self.provider_type == "deepseek" and tool_choice == "required":
                        kwargs["tool_choice"] = "auto"
                    else:
                        kwargs["tool_choice"] = tool_choice

                response = self.client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                message = choice.message

                tool_calls = getattr(message, "tool_calls", None)
                content = getattr(message, "content", "") or ""
                if not tool_calls:
                    # If model returned a substantive statement (e.g. statement or conclusion)
                    if content.strip():
                        return {
                            "name": "statement",
                            "arguments": {"statement": content.strip()},
                            "tool_call_id": None,
                            "raw_message": {"role": "assistant", "content": content},
                            "content": content,
                        }
                    raise ValueError(
                        f"Model did not return any tool_calls. Content: {getattr(message, 'content', '')}"
                    )

                first_call = tool_calls[0]
                func_name = first_call.function.name
                raw_args = first_call.function.arguments

                # Parse JSON arguments
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
