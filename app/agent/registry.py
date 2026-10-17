"""Type-safe tool registry and execution dispatcher for the Vision Sentinel Agent."""

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Raised when an agent tool fails during execution or receives invalid arguments."""

    pass


@dataclass
class ToolDefinition:
    """Represents a registered agent tool with explicit validation schemas."""

    name: str
    description: str
    args_schema: Type[BaseModel]
    handler: Callable[..., Dict[str, Any]]


class ToolRegistry:
    """Registry maintaining tested, validated computer-vision and verification tools.

    Arbitrary shell execution and unrestricted file access are strictly prohibited.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        args_schema: Type[BaseModel],
    ) -> Callable[[Callable[..., Dict[str, Any]]], Callable[..., Dict[str, Any]]]:
        """Decorator to register a tool with explicit argument schema."""

        def decorator(handler: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
            if name in self._tools:
                raise ValueError(f"Tool '{name}' is already registered.")
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                args_schema=args_schema,
                handler=handler,
            )
            return handler

        return decorator

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns metadata and argument schemas for all registered tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema.model_json_schema(),
            }
            for tool in self._tools.values()
        ]

    def execute(
        self, tool_name: str, raw_args: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validates arguments against Pydantic schema and securely executes tool handler."""
        tool = self._tools.get(tool_name)
        if not tool:
            raise ToolExecutionError(f"Tool '{tool_name}' is not registered.")

        try:
            validated_args = tool.args_schema(**raw_args)
        except ValidationError as e:
            raise ToolExecutionError(f"Validation failed for tool '{tool_name}': {e}")

        try:
            # Check if handler accepts context
            sig = inspect.signature(tool.handler)
            if "context" in sig.parameters:
                return tool.handler(validated_args, context=context)
            return tool.handler(validated_args)
        except Exception as e:
            logger.error("Error executing tool '%s': %s", tool_name, e, exc_info=True)
            raise ToolExecutionError(f"Execution error in tool '{tool_name}': {e}")


# Global tool registry singleton
registry = ToolRegistry()
