import os
from prompete import Chat, Prompt, SystemPrompt

from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader, ChoiceLoader

# Note: This example assumes that you have set up your API credentials
# in your environment variables. Depending on the model you choose, you'll need to set:
# - OPENAI_API_KEY for OpenAI models
# - ANTHROPIC_API_KEY for Anthropic models
# - GROQ_API_KEY for Groq models
# For example:
# export OPENAI_API_KEY='your-openai-api-key-here'
# export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
# export GROQ_API_KEY='your-groq-api-key-here'
# Failure to set up these credentials will result in authentication errors
# when trying to use the respective APIs.

# Uncomment the desired model:
#model = "claude-3-haiku-20240307"  # Anthropic model
#model = "llama2-70b-4096"  # Groq model
model = "gpt-4o-mini"  # OpenAI model

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create a Jinja2 Environment instance with multiple template directories
renderer = Environment(
    loader=ChoiceLoader([
        FileSystemLoader(os.path.join(current_dir, "templates")),
    ])
)

# Add a custom filter to the renderer
renderer.filters['uppercase'] = lambda x: x.upper()

@dataclass(frozen=True)
class SpecialSystemPrompt(SystemPrompt):
    language: str

# Create a Chat instance with the renderer
chat = Chat(
    model=model,
    renderer=renderer,
    system_prompt=SpecialSystemPrompt(language="Python")
)

# Define a custom prompt class
@dataclass(frozen=True)
class TaskPrompt(Prompt):
    user_name: str
    language: str
    task: str

# Example conversation
task_prompt = TaskPrompt(
    user_name="Alice",
    language="Python",
    task="write a function to calculate the factorial of a number"
)

# Send the task prompt and print the response
print(chat(task_prompt))

# Follow-up question using a regular string
print(chat("Can you explain how the factorial function works step by step?"))

@dataclass(frozen=True)
class CustomPrompt(Prompt):
    message: str

custom_prompt = CustomPrompt(message="hello world")
print(chat(custom_prompt))
