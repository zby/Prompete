from prompete import Chat

MODEL = "gpt-4o"
#MODEL = "gpt-4o-mini"
#MODEL = "anthropic/claude-3-5-sonnet-latest"

def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """Get the current weather in a given location"""
    # In a real scenario, you would call an actual weather API here
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }

user_question = "My plane is landing in London what should I wear?"

####################
# Getting tool results with LLM interpretation


# Create a Chat instance
chat = Chat(model=MODEL, system_prompt="You are a helpful assistant that uses tools to get information.")

# ask the LLM
content = chat(user_question, tools=[get_current_weather])

# Print the results
print("User question:", user_question)
print()
print("LLM response:", content)
print()
print("Outputs from tools:", chat.get_tool_results())
print()


####################
# Getting tool results without LLM interpretation

# Create a Chat instance
chat = Chat(model=MODEL, system_prompt="You are a helpful assistant that uses tools to get information.")
chat.append(user_question)

message, outputs = chat.complete_once(tools=[get_current_weather])

# message.content should be empty when using OpenAI models - but might be non-empty when using Anthropic models

# Print the results
print("User question:", user_question)
print()
print("LLM response:", content)
print()
print("Outputs from tools:", outputs)
print()

