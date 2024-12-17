from typing import Callable, Optional, Union, Protocol, Any, List
from dataclasses import dataclass, field
import litellm
from pprint import pformat
from datetime import datetime

from llm_easy_tools import get_tool_defs, LLMFunction, ToolResult
from llm_easy_tools.processor import process_message
from llm_easy_tools.types import ChatCompletionMessageToolCall

import logging
import json

#litellm.modify_params = True
#litellm.set_verbose=True

# Configure logging for this module
logger = logging.getLogger("prompete.chat")


@dataclass(frozen=True)
class Prompt:
    def role(self) -> str:
        return "user"


@dataclass(frozen=True)
class SystemPrompt(Prompt):
    """
    System prompt for the chat.
    """

    def role(self) -> str:
        return "system"


class Renderer(Protocol):
    def get_template(self, name: str) -> Any: ...

    def render(self, template: str, **kwargs: Any) -> str: ...

@dataclass
class Message:
    """
    Represents a message in the chat history.
    data is either a litellm.Message (for regular messages), a ToolResult (for tool results) or a dict with role and content.
    """
    data: Union[litellm.Message, ToolResult, dict]
    timestamp: datetime = field(default_factory=datetime.now)

    def make_dict(self) -> dict:
        """Convert the message to a format suitable for LLM input"""
        if isinstance(self.data, ToolResult):
            return self.data.to_message()
        elif isinstance(self.data, dict):
            return self.data
        else:
            return self.data.model_dump()


@dataclass
class Chat:
    model: str
    renderer: Optional[Renderer] = None
    messages: List[Message] = field(default_factory=list)
    system_prompt: Optional[Union[Prompt, str, dict, Message]] = None
    fail_on_tool_error: bool = (
        True  # if False the error message is passed to the LLM to fix the call, if True exception is raised
    )
    one_tool_per_step: bool = (
        True  # for stateful tools executing more than one tool call per step is often confusing for the LLM
    )
    retries: int = 3
    completion_kwargs: dict = field(default_factory=dict)
    run_tools_kwargs: dict = field(default_factory=dict)
    tools: list = field(default_factory=list)
    can_do_response_format: bool = False

    def __post_init__(self):
        if self.system_prompt:
            message = self._make_message(self.system_prompt)
            # this is not perfect - but message should be a dict here
            message.data["role"] = "system"
            self.messages.append(message)
        
        # Check if model supports response_format
        params = litellm.get_supported_openai_params(model=self.model)
        self.can_do_response_format = params and "response_format" in params

    def render_prompt(self, obj: object, **kwargs) -> str:
        template_name = type(obj).__name__
        template = self.renderer.get_template(template_name)

        # Create a context dictionary with the object's public attributes and methods
        obj_context = {
            name: getattr(obj, name) for name in dir(obj) if not name.startswith("_")
        }

        # Merge with kwargs
        obj_context.update(kwargs)

        result = template.render(**obj_context)
        return result

    def _make_message(self, message: Union[Prompt, str, dict, litellm.Message, ToolResult]) -> Message:
        """Convert various message types to a Message object"""
        if isinstance(message, Prompt):
            if self.renderer is None:
                raise ValueError("Renderer is required for Prompt objects")
            content = self.render_prompt(message)
            return Message(data={"role": message.role(), "content": content.strip()})
        elif isinstance(message, str):
            return Message(data={"role": "user", "content": message})
        elif isinstance(message, dict):
            if "role" not in message or "content" not in message:
                raise ValueError("Dict message must contain 'role' and 'content' keys")
            return Message(data=message)
        elif isinstance(message, litellm.Message) or isinstance(message, ToolResult):
            return Message(data=message)
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    def append(self, content: Union[Prompt, str, dict, litellm.Message, ToolResult]) -> None:
        """
        Append a message to the chat.
        """
        message = self._make_message(content)
        self.messages.append(message)

    def complete_once(
        self,
        tools: list = [],
        **kwargs
    ) -> tuple[litellm.Message, Optional[List[Any]]]:
        """
        Performs one step of LLM interaction: gets a response and processes any tool calls.

        Args:
            tools:
            **kwargs: Additional kwargs for get_llm_response and process methods

        Returns:
            tuple containing:
            - The LLM response message
            - List of tool outputs if any tools were called, None otherwise
        """
        # Add any new tools to the list
        for tool in tools:
            self._add_tool(tool)

        response = self._get_llm_response(**kwargs)

        outputs = None
        if self._is_tool_calls_message(response):
            outputs = self._run_tools(response)

        return response, outputs

    def __call__(
        self,
        message: Prompt | dict | litellm.Message | str,
        max_llm_requests: int = 2,
        **kwargs
    ) -> Optional[str]:

        logging.debug(f"Starting chat call with message: {message}")
        self.append(message)

        req_count = 1
        while req_count <= max_llm_requests:
            if req_count == max_llm_requests:
                # we don't want tool calls at the last completion
                kwargs['tool_choice'] = 'none'

            response, outputs = self.complete_once(**kwargs)

            # If no tool calls were made, return the response content
            if outputs is None:
                logging.debug(f"Found response without tool calls after {req_count} loops")
                return response.content

            logging.debug(f"Processing tool calls, loop {req_count + 1}")
            req_count += 1

        if max_llm_requests > 1:
            logging.warning(f"Reached maximum loops ({max_llm_requests}) without finding non-tool response")
        return None

    def messages_to_dict_list(self) -> List[dict]:
        """Convert the messages list to a format suitable for LLM input"""
        return [msg.make_dict() for msg in self.messages]

    def _add_tool(self, tool: Callable) -> None:
        if tool not in self.tools:
            self.tools.append(tool)

    def _get_llm_response(self, strict=False, response_format=None, **kwargs) -> litellm.Message:
        if strict and not self.tools:
            raise ValueError("Tools must be provided if strict is True")

        if response_format:
            if self.can_do_response_format:
                kwargs["response_format"] = response_format
            else:
                self._add_tool(response_format)
                kwargs["tool_choice"] = response_format.__name__

        schemas = get_tool_defs(self.tools, strict=strict)
        args = {
            "model": self.model,
            "messages": self.messages_to_dict_list(),
            "num_retries": self.retries,
        }
        if self.completion_kwargs:
            args.update(self.completion_kwargs)

        if len(schemas) > 0:
            args["tools"] = schemas
            if not args.get("tool_choice"):
                args["tool_choice"] = "auto"

        args.update(kwargs)

        logger.debug(f"llm_reply args: {pformat(args, width=120)}")

        result = litellm.completion(**args)

        logger.debug(
            f"Received response from LLM: {pformat(result.to_dict(), width=120)}"
        )

        message = result.choices[0].message

        if (
            self.one_tool_per_step
            and self._is_tool_calls_message(message)
            and message.tool_calls
        ):
            if len(message.tool_calls) > 1:
                logging.warning(f"More than one tool call: {message.tool_calls}")
                message.tool_calls = [message.tool_calls[0]]

        if response_format:
            if not self.can_do_response_format:
                outputs = self._run_tools(message)
                message.content = outputs[0]
                message.tool_calls = None
            else:
                string_content = message.content
                message.content = response_format.model_validate_json(string_content)

        self.append(message)

        return message

    def _run_tools(self, message: litellm.Message):
        kwargs = self.run_tools_kwargs
        results = process_message(message, self.tools, **kwargs)
        outputs = []
        for result in results:
            if result.soft_errors:
                for soft_error in result.soft_errors:
                    logger.warning(soft_error)
            self.append(result)
            if result.error and self.fail_on_tool_error:
                print(result.stack_trace)
                raise Exception(result.error)
            if isinstance(result.output, Prompt):
                # TODO: This is not consistent
                #  the messaeg saved in the chat is not rendered but converted to a string in LLMEasyTools
                output = self.render_prompt(result.output)
                outputs.append(output)
            else:
                outputs.append(result.output)

        return outputs

    def _is_tool_calls_message(self, message: Message) -> bool:
        if isinstance(message, litellm.Message):
            return hasattr(message, "tool_calls") and message.tool_calls
        return False

    def _get_last_tool_calls_message(self) -> Optional[litellm.Message]:
        """
        Return the last message in the chat history if it has 'tool_calls' key, or None if the history is empty.
        """
        for message in reversed(self.messages):
            if isinstance(message.data, litellm.Message):
                if hasattr(message.data, "tool_calls") and message.data.tool_calls:
                    return message.data
        return None

    def get_tool_results(self) -> List[ToolResult]:
        """
        Finds ToolResults that correspond to the tool calls in the last tool_calls message.
        Returns an empty list if there are no tool calls or no matching results.
        """
        tool_calls_msg = self._get_last_tool_calls_message()
        if not tool_calls_msg:
            return []

        # Get the IDs from the tool calls
        tool_call_ids = [call.id for call in tool_calls_msg.tool_calls]

        # Find ToolResults that match these IDs
        return [
            msg.data for msg in self.messages
            if isinstance(msg.data, ToolResult)
            and msg.data.tool_call_id in tool_call_ids
        ]


if __name__ == "__main__":
    import os
    from jinja2 import Environment, DictLoader, FileSystemLoader, ChoiceLoader
    from pprint import pprint

    # Create a simple Chat example without a renderer
    simple_chat = Chat(model="gpt-3.5-turbo")

    # Create a simple message
    simple_message = "Hello, AI!"

    # Use make_message and print the result
    print("Simple Chat Example:")
    print(simple_chat._make_message(simple_message).make_dict())

    print("\n" + "=" * 50 + "\n")

    @dataclass(frozen=True)
    class AssistantPrompt(Prompt):
        answer: str

        def role(self) -> str:
            return "assistant"

    @dataclass(frozen=True)
    class SpecialPrompt(Prompt):
        content: str

        def render(self):
            return f"Special prompt: {self.content.upper()}"

    @dataclass(frozen=True)
    class Prompt1(Prompt):
        value: str

    @dataclass(frozen=True)
    class Prompt2(Prompt):
        value: str

    # Create the renderer
    templates = {
        "SystemPrompt": "You are a helpful assistant.",
        "AssistantPrompt": "Assistant: {{answer}}",
        "SpecialPrompt": "{{__str__()}}",
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_dirs = [
        os.path.join(current_dir, "test_data", "prompts1"),
        os.path.join(current_dir, "test_data", "prompts2"),
    ]

    renderer = Environment(
        loader=ChoiceLoader([DictLoader(templates), FileSystemLoader(template_dirs)])
    )

    # Create Chat with the separate renderer
    chat = Chat(model="gpt-3.5-turbo", renderer=renderer)

    # Create example prompts
    prompt1 = Prompt1(value="Example1")
    prompt2 = Prompt2(value="Example2")
    assistant_prompt = AssistantPrompt(answer="This is an assistant response.")

    # Add prompts to the chat
    pprint(chat._make_message(prompt1).make_dict())
    pprint(chat._make_message(prompt2).make_dict())
    pprint(chat._make_message(assistant_prompt).make_dict())

    # This does ot work!!!
#    @dataclass(frozen=True)
#    class TestPrompt(Prompt):
#        role: str
#
#    test_prompt = TestPrompt(role="some role")
#    try:
#        chat.make_message(test_prompt)
#    except ValueError as e:
#        print(f"Error message: {str(e)}")
# from hello import hello
