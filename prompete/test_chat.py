import pytest
from dataclasses import dataclass
import litellm
#from litellm import Message, TextCompletionResponse, TextChoices
from typing import Any, Optional
import json

from llm_easy_tools import ToolResult
from jinja2 import Environment, DictLoader, FileSystemLoader, ChoiceLoader

from prompete import Chat, Prompt, SystemPrompt, Message

def create_mock_response(
    content: Any, role: str = "assistant", tool_calls: Optional[list] = None
) -> litellm.TextCompletionResponse:
    message = litellm.Message(
        content=json.dumps(content) if isinstance(content, dict) else content,
        role=role,
        tool_calls=tool_calls or [],
    )
    return litellm.TextCompletionResponse(choices=[litellm.TextChoices(message=message)])


def test_append():
    @dataclass(frozen=True)
    class GreetingPrompt(Prompt):
        name: str
        time_of_day: str

        @property
        def hello(self):
            return "Hello"

    templates = {
        "GreetingPrompt": "{{hello}} {{name}}! Good {{time_of_day}}.",
    }
    renderer = Environment(loader=DictLoader(templates))
    chat = Chat(model="gpt-3.5-turbo", renderer=renderer)

    greeting = GreetingPrompt(name="Alice", time_of_day="morning")
    chat.append(greeting)

    assert len(chat.messages) == 1
    assert isinstance(chat.messages[0], Message)
    message_dict = chat.messages[0].make_dict()
    assert message_dict["role"] == "user"
    assert message_dict["content"] == "Hello Alice! Good morning."

    chat.append("Hello, can you help me?")
    assert len(chat.messages) == 2
    assert isinstance(chat.messages[1], Message)
    message_dict = chat.messages[1].make_dict()
    assert message_dict["role"] == "user"
    assert message_dict["content"] == "Hello, can you help me?"


def test_system_prompt():
    chat = Chat(model="gpt-3.5-turbo", system_prompt="You are a helpful AI assistant.")

    assert len(chat.messages) == 1
    message_dict = chat.messages[0].make_dict()
    assert message_dict["role"] == "system"
    assert message_dict["content"] == "You are a helpful AI assistant."

    @dataclass(frozen=True)
    class SpecialSystemPrompt(SystemPrompt):
        capabilities: str

    templates = {
        "SpecialSystemPrompt": "You are a helpful AI assistant. Your capabilities include: {{capabilities}}."
    }
    renderer = Environment(loader=DictLoader(templates))

    system_prompt = SpecialSystemPrompt(
        capabilities="answering questions, providing information, and assisting with tasks"
    )

    chat = Chat(model="gpt-3.5-turbo", renderer=renderer, system_prompt=system_prompt)

    assert len(chat.messages) == 1
    message_dict = chat.messages[0].make_dict()
    assert message_dict["role"] == "system"
    assert message_dict["content"] == (
        "You are a helpful AI assistant. Your capabilities include: "
        "answering questions, providing information, and assisting with tasks."
    )

    # Test adding a user message after system prompt
    chat.append({"role": "user", "content": "Hello, can you help me?"})

    assert len(chat.messages) == 2
    message_dict = chat.messages[1].make_dict()
    assert message_dict["role"] == "user"
    assert message_dict["content"] == "Hello, can you help me?"


def test_chat_append_tool_result():
    chat = Chat(model="gpt-3.5-turbo")

    # Create a ToolResult
    tool_result = ToolResult(
        tool_call_id="123",
        name="TestTool",
        output="This is the result of the test tool.",
    )

    # Append the ToolResult to the chat
    chat.append(tool_result.to_message())

    # Assert that the message was added correctly
    assert len(chat.messages) == 1
    message_dict = chat.messages[0].make_dict()
    assert message_dict["role"] == "tool"
    assert message_dict["name"] == "TestTool"
    assert message_dict["content"] == "This is the result of the test tool."


def test_chat_without_renderer():
    chat = Chat(model="gpt-3.5-turbo")

    chat.append("Hello, AI!")
    assert len(chat.messages) == 1
    message_dict = chat.messages[0].make_dict()
    assert message_dict["role"] == "user"
    assert message_dict["content"] == "Hello, AI!"

    chat.append({"role": "assistant", "content": "Hello! How can I assist you today?"})
    assert len(chat.messages) == 2
    message_dict = chat.messages[1].make_dict()
    assert message_dict["role"] == "assistant"
    assert message_dict["content"] == "Hello! How can I assist you today?"

    with pytest.raises(ValueError):

        @dataclass(frozen=True)
        class TestPrompt(Prompt):
            value: str

        chat.append(TestPrompt(value="test"))


def test_chat_with_renderer():
    templates = {
        "TestPrompt": "This is a test prompt with {{value}}.",
    }
    renderer = Environment(loader=DictLoader(templates))
    chat = Chat(model="gpt-3.5-turbo", renderer=renderer)

    @dataclass(frozen=True)
    class TestPrompt(Prompt):
        value: str

    chat.append(TestPrompt(value="example"))
    assert len(chat.messages) == 1
    message_dict = chat.messages[0].make_dict()
    assert message_dict["role"] == "user"
    assert message_dict["content"] == "This is a test prompt with example."


def test_invalid_message_type():
    chat = Chat(model="gpt-3.5-turbo")

    with pytest.raises(ValueError):
        chat.append(123)  # Invalid message type


def test_invalid_dict_message():
    chat = Chat(model="gpt-3.5-turbo")

    with pytest.raises(ValueError):
        chat.append({"invalid": "message"})  # Dict without 'role' and 'content'


def test_template_loading():
    # Create a Chat instance with templates_dirs
    renderer = Environment(
        loader=FileSystemLoader(
            ["prompete/test_data/prompts1", "prompete/test_data/prompts2"]
        )
    )
    chat = Chat(model="gpt-3.5-turbo", renderer=renderer)

    # Check if the templates were loaded correctly
    t = chat.renderer.get_template("Prompt1")
    assert (
        t.render({"value": "test"})
        == 'This is Prompt1 from prompts1\nSome value: "test"'
    )
    t = chat.renderer.get_template("Prompt2")
    assert t.render({"value": "test"}) == 'This is Prompt2.\nSome value: "test"'

    # Create a Chat instance with prompts2 first
    renderer = Environment(
        loader=FileSystemLoader(
            ["prompete/test_data/prompts2", "prompete/test_data/prompts1"]
        )
    )
    chat = Chat(model="gpt-3.5-turbo", renderer=renderer)

    # Check if the templates were loaded correctly
    t = chat.renderer.get_template("Prompt1")
    assert t.render({}) == "This is Prompt1 from prompts2."
    t = chat.renderer.get_template("Prompt2")
    assert t.render({"value": "test"}) == 'This is Prompt2.\nSome value: "test"'


def test_renderer_with_templates_dict():
    # Create a Chat instance with templates dictionary and a template directory
    templates = {
        "CustomPrompt1": "This is a custom prompt: {{value}}",
        "CustomPrompt2": "Another custom prompt: {{name}}",
        "Prompt1": "Overridden Prompt1: {{value}}",  # This should override the one from disk
    }
    renderer = Environment(
        loader=ChoiceLoader(
            [DictLoader(templates), FileSystemLoader(["prompete/test_data/prompts1"])]
        )
    )
    chat = Chat(model="gpt-3.5-turbo", renderer=renderer)

    # Check if the templates were loaded correctly
    t1 = chat.renderer.get_template("CustomPrompt1")
    assert t1.render({"value": "test"}) == "This is a custom prompt: test"

    t2 = chat.renderer.get_template("CustomPrompt2")
    assert t2.render({"name": "John"}) == "Another custom prompt: John"

    # Test that the template from disk is loaded
    t3 = chat.renderer.get_template("Prompt2")
    assert t3.render({"value": "test"}) == 'This is Prompt2.\nSome value: "test"'

    # Test that Prompt1 is overridden
    t4 = chat.renderer.get_template("Prompt1")
    assert t4.render({"value": "test"}) == "Overridden Prompt1: test"


def test_chat_with_custom_environment():
    # Create a custom Jinja2 Environment
    custom_templates = {
        "CustomPrompt": "This is a custom prompt with {{value}}",
    }
    custom_env = Environment(loader=DictLoader(custom_templates))

    # Create a Chat instance with the custom environment
    chat = Chat(model="gpt-3.5-turbo", renderer=custom_env)

    # Test if the custom template is accessible
    t = chat.renderer.get_template("CustomPrompt")
    assert t.render({"value": "test"}) == "This is a custom prompt with test"


def test_chat_with_environment_extension():
    # Create a custom Jinja2 Environment with an extension
    custom_templates = {
        "UppercasePrompt": "This prompt uses a custom filter: {{ value | custom_upper }}",
    }
    custom_env = Environment(
        loader=DictLoader(custom_templates),
    )
    custom_env.filters["custom_upper"] = lambda x: x.upper()

    # Create a Chat instance with the custom environment
    chat = Chat(model="gpt-3.5-turbo", renderer=custom_env)

    # Test if the custom template with the extension is working
    t = chat.renderer.get_template("UppercasePrompt")
    assert t.render({"value": "test"}) == "This prompt uses a custom filter: TEST"

    # Ensure that the custom filter is available in render_prompt
    @dataclass(frozen=True)
    class UppercasePrompt(Prompt):
        value: str

    prompt = UppercasePrompt(value="hello")
    rendered = chat.render_prompt(prompt)
    assert rendered == "This prompt uses a custom filter: HELLO"


def test_render_prompt():
    renderer = Environment(
        loader=DictLoader({"TwoValsPrompt": "value1: {{value1}}\nvalue2: {{value2}}"})
    )
    chat = Chat(
        model="gpt-3.5-turbo",
        renderer=renderer,
    )

    @dataclass(frozen=True)
    class TwoValsPrompt(Prompt):
        value1: str

    prompt1 = TwoValsPrompt(value1="test1")

    assert chat.render_prompt(prompt1) == "value1: test1\nvalue2: "

    # Check kwargs
    assert chat.render_prompt(prompt1, value2="test2") == "value1: test1\nvalue2: test2"


def test_process_tool_calls(mocker):
    def get_current_weather(location: str, unit: str = "celsius") -> str:
        """Get the current weather in a given location"""
        weather_data = {
            "location": location,
            "temperature": 22,
            "unit": unit,
            "forecast": ["sunny", "windy"],
        }
        return weather_data

    weather_args = {"location": "London", "unit": "celsius"}
    ultimate_answer = "The weather in London is sunny and 22°C"

        # Mock completion - first call returns tool call, second returns final response
    mock_completion = mocker.patch("litellm.completion")
    mock_completion.side_effect = [
        create_mock_response(
            content=None,
            tool_calls=[{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": json.dumps(weather_args),
                    },
            }]
        ),
        create_mock_response(ultimate_answer),
        create_mock_response('addtional tool call'),
    ]

    chat = Chat(model="gpt-4-0125-preview", tools=[get_current_weather], max_loops=2)

    # Call chat with user question
    user_question = "What's the weather like in London?"
    content = chat(user_question)

    # Check completion was called twice
    assert mock_completion.call_count == 2

    # Check final response
    assert content == ultimate_answer
    
    # Verify messages sequence
    assert len(chat.messages) == 4
    message_dict = chat.messages[0].make_dict()
    assert message_dict["role"] == "user"
    assert message_dict["content"] == user_question
    message_dict = chat.messages[1].make_dict()
    assert message_dict["role"] == "assistant"
    assert message_dict["tool_calls"][0]["function"]["name"] == "get_current_weather"
    message_dict = chat.messages[2].make_dict()
    assert message_dict["role"] == "tool"
    assert message_dict["content"] == str(get_current_weather(**weather_args))
    message_dict = chat.messages[3].make_dict()
    assert message_dict["role"] == "assistant"
    assert message_dict["content"] == "The weather in London is sunny and 22°C"

    # Verify first call had tool_choice='auto'
    first_call_args = mock_completion.call_args_list[0][1]
    assert first_call_args["tool_choice"] == "auto"

    # Verify that tool_choice passes to the completion function
    content = chat(user_question, tool_choice="none")
    call_args = mock_completion.call_args_list[2][1]
    assert call_args["tool_choice"] == "none"


def test_chat_response_format(mocker):
    from pydantic import BaseModel

    class TestResponseFormat(BaseModel):
        message: str
        confidence: float

    test_response_object = TestResponseFormat(message="Test response", confidence=0.95)

    # Mock the completion function
    mock_completion = mocker.patch("litellm.completion")
    mock_completion.return_value = create_mock_response(
        test_response_object.model_dump()
    )

    # Create a Chat instance
    chat = Chat(model="some_model")
    chat.can_do_response_format = True

    # Call the chat with response_format
    response = chat(
        "Hello, can you give me a test response?", response_format=TestResponseFormat
    )

    # Assert that the response is an instance of TestResponseFormat
    assert isinstance(response, TestResponseFormat)
    assert response.message == "Test response"
    assert response.confidence == 0.95

    # Verify that the completion function was called with the correct parameters
    mock_completion.assert_called_once()
    call_args = mock_completion.call_args[1]
    assert call_args["response_format"] == TestResponseFormat

    # Test with an invalid response
    mock_completion.return_value = create_mock_response({"message": "Invalid response"})

    # This should raise a ValidationError
    with pytest.raises(ValueError):
        chat("hello", response_format=TestResponseFormat)
        # TODO: this generates a warning: "Expected `str` but got `test_chat_response_format.<locals>.TestResponseFormat` with..."


def test_chat_emulate_response_format(mocker):
    from pydantic import BaseModel

    class TestResponseFormat(BaseModel):
        message: str
        confidence: float

    test_response_object = TestResponseFormat(
        message="Emulated response", confidence=0.8
    )

    # Mock the completion function
    mock_completion = mocker.patch("litellm.completion")
    mock_completion.return_value = create_mock_response(
        content=None,
        tool_calls=[
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "TestResponseFormat",
                    "arguments": json.dumps(test_response_object.model_dump()),
                },
            }
        ],
    )

    # Create a Chat instance with emulate_response_format=True
    chat = Chat(model="some_model")

    # Call the chat with response_format
    response = chat(
        "Hello, can you give me a test response?", response_format=TestResponseFormat
    )

    # Assert that the response is an instance of TestResponseFormat
    assert isinstance(response, TestResponseFormat)
    assert response == test_response_object

    # Verify that the completion function was called with the correct parameters
    mock_completion.assert_called_once()
    call_args = mock_completion.call_args[1]
    assert "response_format" not in call_args
    assert len(call_args["tools"]) == 1
    assert call_args["tools"][0]["function"]["name"] == "TestResponseFormat"
