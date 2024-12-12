from typing import Callable, Optional, Union, Protocol, Any
from dataclasses import dataclass, field
import litellm
from pprint import pformat

from llm_easy_tools import get_tool_defs, LLMFunction
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
class Chat:
    model: str
    renderer: Optional[Renderer] = None
    messages: list[dict] = field(default_factory=list)
    system_prompt: Optional[Union[Prompt, str, dict, litellm.Message]] = None
    fail_on_tool_error: bool = (
        True  # if False the error message is passed to the LLM to fix the call, if True exception is raised
    )
    one_tool_per_step: bool = (
        True  # for stateful tools executing more than one tool call per step is often confusing for the LLM
    )
    max_loops: int = 3
    retries: int = 3
    custom_llm_provider: Optional[str] = None
    tools: list = field(default_factory=list)
    can_do_response_format: bool = False

    def __post_init__(self):
        if self.system_prompt:
            message = self.make_message(self.system_prompt)
            message["role"] = "system"
            self.append(message)
        
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

    def make_message(self, message: Union[Prompt, str, dict, litellm.Message]) -> dict:
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
        elif isinstance(message, litellm.Message):
            return message.model_dump()
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    def append(self, message: Union[Prompt, str, dict, litellm.Message]) -> None:
        """
        Append a message to the chat.
        """
        message_dict = self.make_message(message)
        logging.debug(f"Appending message: {message_dict}")
        self.messages.append(message_dict)

    def __call__(
        self,
        message: Prompt | dict | litellm.Message | str,
        response_format=None,
        tools: Optional[list] = None,
        **kwargs
    ) -> str:
        if response_format:
            if self.can_do_response_format:
                kwargs["response_format"] = response_format
            else:
                if tools:
                    raise ValueError("When emulating response_format you cannot have tools")
                tools = [response_format]

        # Add any new tools to the list
        if tools:
            for tool in tools:
                if tool not in self.tools:
                    self.tools.append(tool)

        logging.debug(f"Starting chat call with message: {message}")
        self.append(message)

        loop_count = 0
        while loop_count < self.max_loops:
            response_content = self.get_llm_response(**kwargs)

            # Check if response has tool calls
            if not self.get_tool_calls_message():
                logging.debug(f"Found response without tool calls after {loop_count} loops")
                return response_content

            # Process tool calls and continue loop
            logging.debug(f"Processing tool calls, loop {loop_count + 1}")
            self.process()
            loop_count += 1

        logging.warning(f"Reached maximum loops ({self.max_loops}) without finding non-tool response")
        response_content = self.get_llm_response(**kwargs)
        return response_content

    def get_llm_response(self, strict=False, **kwargs) -> str:
        if strict and not self.tools:
            raise ValueError("Tools must be provided if strict is True")
        schemas = get_tool_defs(self.tools, strict=strict)
        args = {
            "model": self.model,
            "messages": self.messages,
            "num_retries": self.retries,
        }
        if self.custom_llm_provider:
            args["custom_llm_provider"] = self.custom_llm_provider

        if len(schemas) > 0:
            args["tools"] = schemas
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
            and self._is_tool_call_message(message)
            and message.tool_calls
        ):
            if len(message.tool_calls) > 1:
                logging.warning(f"More than one tool call: {message.tool_calls}")
                message.tool_calls = [message.tool_calls[0]]

        self.append(message)

        return message.content

    def process(self, **kwargs):
        message = self.get_tool_calls_message()
        if not message:
            raise ValueError("No message to process")
        results = process_message(message, self.tools, **kwargs)
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

    def _is_tool_call_message(self, message: litellm.Message) -> bool:
        return hasattr(message, "tool_calls") and message.tool_calls

    def get_tool_calls_message(self) -> litellm.Message:
        """
        Return the last message in the chat history if it has 'tool_calls' key, or None if the history is empty.
        """
        if not self.messages:
            return None
        dict_message = self.messages[-1]
        message = litellm.Message(**dict_message)
        if self._is_tool_call_message(message):
            return message
        else:
            return None
    

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
