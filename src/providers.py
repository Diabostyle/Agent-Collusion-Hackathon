import json
import logging
import time
from typing import Any, Dict, List, Optional

import litellm

# Reduce verbose litellm logs
litellm.suppress_debug_info = True
logger = logging.getLogger("AgentProvider")


class LLMToolCallError(RuntimeError):
    """Raised when the LLM fails to output a valid tool call after max retries."""
    pass


class AgentProvider:
    """Wrapper around LLM completion API with automatic retries and JSON schema enforcement."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.5,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    def generate_action(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Any = "required",
    ) -> Dict[str, Any]:
        """Queries the model and returns a validated tool call.

        Returns:
            dict containing:
                - "name": str (name of the tool called)
                - "arguments": dict (parsed arguments)
                - "raw_tool_call": dict (original tool call payload)
                - "raw_message": dict (full message dictionary from LLM)
                - "content": Optional[str] (agent thoughts / prose, if any)
        """
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice=tool_choice if tools else None,
                    temperature=self.temperature,
                )

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
