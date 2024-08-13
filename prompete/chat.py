from typing import Callable, Optional, Union, Protocol, Any
from dataclasses import dataclass, field
from litellm import completion, ModelResponse, Message
from pprint import pformat

from llm_easy_tools import get_tool_defs, LLMFunction
from llm_easy_tools.processor import process_message

import logging


# Configure logging for this module
logger = logging.getLogger("answerbot.chat")
logger.setLevel(logging.DEBUG)  # Set the logger to capture DEBUG level messages


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
class Chat:
    model: str
    renderer: Optional[Renderer] = None
    messages: list[Union[dict, Message]] = field(default_factory=list)
    system_prompt: Optional[Union[Prompt, str, dict, Message]] = None
    fail_on_tool_error: bool = (
        True  # if False the error message is passed to the LLM to fix the call, if True exception is raised
    )
    one_tool_per_step: bool = (
        True  # for stateful tools executing more than one tool call per step is often confusing for the LLM
    )
    saved_tools: list[Union[LLMFunction, Callable]] = field(default_factory=list)
    retries: int = 3
    custom_llm_provider: Optional[str] = None

    def __post_init__(self):
        if self.system_prompt:
            message = self.make_message(self.system_prompt)
            message["role"] = "system"
            self.append(message)

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

    def make_message(self, message: Union[Prompt, str, dict, Message]) -> dict:
        if isinstance(message, Prompt):
            if self.renderer is None:
                raise ValueError("Renderer is required for Prompt objects")
            content = self.render_prompt(message)
            return {"role": message.role(), "content": content.strip()}
        elif isinstance(message, str):
            return {"role": "user", "content": message}
        elif isinstance(message, dict):
            if "role" not in message or "content" not in message:
                raise ValueError("Dict message must contain 'role' and 'content' keys")
            return message
        elif isinstance(message, Message):
            return message
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    def append(self, message: Union[Prompt, str, dict, Message]) -> None:
        """
        Append a message to the chat.
        """
        message_dict = self.make_message(message)
        self.messages.append(message_dict)

    def __call__(self, message: Prompt | dict | Message | str, **kwargs) -> str:
        """
        Allow the Chat object to be called as a function.
        Appends the given message and calls llm_reply with the provided kwargs.
        Returns the content of the response message as a string.
        """
        self.append(message)
        response = self.llm_reply(**kwargs)
        return response.choices[0].message.content

    def llm_reply(self, tools=[], strict=False, **kwargs) -> ModelResponse:
        if strict and not tools:
            raise ValueError("Tools must be provided if strict is True")
        self.saved_tools = tools
        schemas = get_tool_defs(tools, strict=strict)
        args = {
            "model": self.model,
            "messages": self.messages,
            "num_retries": self.retries,
        }
        if self.custom_llm_provider:
            args["custom_llm_provider"] = self.custom_llm_provider

        if len(schemas) > 0:
            args["tools"] = schemas
            if len(schemas) == 1:
                args["tool_choice"] = {
                    "type": "function",
                    "function": {"name": schemas[0]["function"]["name"]},
                }
            else:
                args["tool_choice"] = "auto"

        args.update(kwargs)

        logger.debug(f"llm_reply args: {pformat(args, width=120)}")
        logger.debug(f"Sending request to LLM with {len(self.messages)} messages")

        result = completion(**args)

        logger.debug(
            f"Received response from LLM: {pformat(result.to_dict(), width=120)}"
        )

        message = result.choices[0].message

        if (
            self.one_tool_per_step
            and hasattr(message, "tool_calls")
            and message.tool_calls
        ):
            if len(message.tool_calls) > 1:
                logging.warning(f"More than one tool call: {message.tool_calls}")
                message.tool_calls = [message.tool_calls[0]]

        if len(schemas) > 0:
            if not hasattr(message, "tool_calls") or not message.tool_calls:
                logging.warning("No function call.")

        self.append(message)

        return result

    def process(self, **kwargs):
        if not self.messages:
            raise ValueError("No messages to process")
        message = self.messages[-1]
        results = process_message(message, self.saved_tools, **kwargs)
        outputs = []
        for result in results:
            if result.soft_errors:
                for soft_error in result.soft_errors:
                    logger.warning(soft_error)
            self.append(result.to_message())
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

    def get_last_message(self) -> Optional[Union[dict, Message]]:
        """
        Return the last message in the chat history, or None if the history is empty.
        """
        return self.messages[-1] if self.messages else None


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
    print(simple_chat.make_message(simple_message))

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
    pprint(chat.make_message(prompt1))
    pprint(chat.make_message(prompt2))
    pprint(chat.make_message(assistant_prompt))

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
