# Prompete

Prompete is a wrapper for LiteLLM.
It integrates LLMEasyTools, and Jinja2 templates to create a flexible system for
managing prompts and chat interactions. Part of the API is inspired by Claudette.


The key idea behind Prompete is that LLM prompts contain two distinct components:
- data: the information that the LLM is supposed to manipulate
- instructions: the instructions on how to interpret and use that data

These two parts have different nature, and are manipulated differently.
They change in different rythms and in different phases of the project. They are
edited in different ways and mixing them together is error prone and makes the
prompt management harder. For example long text blocks in code demolish the
visual clues a programmer relies on when reading that code.

Prompete tries to address this by separating the two components into data structures and
templates.

But that does not mean that when using Prompete you need to start immediately with 
templated prompts. You can start working in Prompete with simple string prompts,
then progressively adopt more advanced features as your needs evolve.

## Features

- Template-based prompt generation using Jinja2
- Integration with various LLM APIs through LiteLLM
- Emulate `response_format` by using `tools` (for models that don't support `response_format`)
- Easy function calling with LLMEasyTools
- Conversation management with the Chat interface
- System prompts and custom prompt roles

## Installation

Install Prompete using pip:

```bash
pip install prompete
```

## Quick Start

First you need to set up your API credentials in your environment variables.
Depending on the model you choose, you'll need to set:
- OPENAI_API_KEY for OpenAI models
- ANTHROPIC_API_KEY for Anthropic models

For example:
```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

### Basic example
# text prompt

```python
from prompete import Chat

model = "gpt-4o-mini"  # OpenAI model

# Create a Chat instance with a system prompt
chat = Chat(
    model=model,
    system_prompt="You are a helpful assistant specializing in Python programming."
)

# Start the conversation
user_message = "What's the difference between a list and a tuple in Python?"
response = chat(user_message)

print("User:", user_message)
print("AI:", response)
```

### Templating example

A full example of templating with Prompete can be found in the `examples/templating.py` file.

To use templating with Prompete, you need to create a Jinja2 Environment.
You can use all features of Jinja2 like multiple directories to search for template files, add
custom filters, etc.

```python
# Create a Jinja2 Environment instance with multiple template directories
renderer = Environment(
    loader=ChoiceLoader([
        FileSystemLoader(os.path.join(current_dir, "templates")),
    ])
)
```

The data is passed to the template as a subclass of `Prompt`.

```python
@dataclass(frozen=True)
class TaskPrompt(Prompt):
    user_name: str
    language: str
    task: str

task_prompt = TaskPrompt(
    user_name="Alice",
    language="Python",
    task="write a function to calculate the factorial of a number"
)

print(chat(task_prompt))
```

The prompt tempalte is found by looking up the class name in the templates defined in the renderer.
The template can use the prompt fields as variables.

### Function Calling

Prompete integrates with LLMEasyTools to provide function calling capabilities.
Here's how it works:

#### Basic Function Calling

```python
from prompete import Chat

def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get the current weather in a given location"""
    # Simulate weather API call
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }

chat = Chat(model="gpt-4")
response = chat("Should I bring an umbrella to London today?", tools=[get_weather])
print(tool_results)
```

#### Two Ways to Handle Tool Results

When working with function calls, there are two main approaches to handle the results:

1. **LLM Interpretation**: Let the LLM interpret the tool results and provide a human-friendly response
```python
# LLM will call the function and interpret results
chat = Chat(model="gpt-4")
response = chat("Should I bring an umbrella to London today?", tools=[get_weather])
print(response)  # LLM provides a natural language response based on the weather data
```
The pitfall of this more 'agentic' approach is that the LLM can sometimes decide
to guess the weather instead of calling the function.

2. **Direct Code Processing**: Handle the tool results directly in your code
```python
# Get direct access to both LLM response and tool results
chat = Chat(model="gpt-4")
response, tool_results = chat.complete_once("What's the weather in London?", tools=[get_weather])
if tool_results:
    weather_data = tool_results[0]
    print(f"Temperature: {weather_data['temperature']}°{weather_data['unit']}")
```

#### Control Over Tool Execution

Prompete provides two main ways to control tool execution:

- `max_loops`: - field in the Chat object - limits how many LLM request can be made in a single conversation turn
- `tool_choice`: - parameter passed to LiteLLM - you can use it to force the LLM to use a specific tool

```python
# Example with controlled tool execution
chat = Chat(
    model="gpt-4",
    max_loops=2,  # Allow up to 2 LLM calls per turn
    one_tool_per_step=True,  # Prevent multiple tool calls at once
    fail_on_tool_error=False  # Let LLM handle any tool errors
)

# Force the use of get_weather tool
response, results = chat.complete_once(
    "How's the weather?", 
    tools=[get_weather],
    tool_choice="get_weather"
)
```

#### Tool Results in Conversation History

All tool calls and their results are automatically saved in the chat history, making them available for context in future conversation turns.

## Key Concepts

- **Chat**: The main class for managing conversations and interacting with LLMs.
- **Prompt**: Base class for creating custom prompt types.
- **renderer**: Jinja2 Environment for rendering prompts with dynamic content.

## Contributing

We welcome contributions to Prompete! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and write tests if applicable
4. Submit a pull request with a clear description of your changes

For major changes, please open an issue first to discuss the proposed changes.

### Writing Test Cases

We strongly encourage writing test cases for both bug reports and feature requests:

- For bugs: Include a test case that reproduces the issue, showing expected vs. actual behavior.
- For features: Provide test cases describing the desired functionality, including inputs and expected outputs.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/zby/prompete/issues) on our GitHub repository.

## Acknowledgements

Prompete is built upon several excellent libraries:

- [LiteLLM](https://github.com/BerriAI/litellm) for universal LLM API support
- [Jinja2](https://jinja.palletsprojects.com/) for powerful templating capabilities
- [Claudette](https://claudette.answer.ai/) for the API inspiration

We're grateful to the maintainers and contributors of these projects.
