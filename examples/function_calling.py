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


# Create a Chat instance
chat = Chat(model=MODEL, max_loops=1, system_prompt="You are a helpful assistant that uses tools to get information.")

# Define the user's question
user_question = "My plane is landing in London what should I wear?"
content = chat(user_question, tools=[get_current_weather])

# Print the results
print("User:", user_question)
print("Content of the response:", content)
print("Outputs from tools:", chat.get_tool_results())
print()
print()

# A Chat instance that processes the tool results
chat = Chat(model=MODEL, max_loops=2, system_prompt="You are a helpful assistant that uses tools to get information.")

# Define the user's question
user_question = "My plane is landing in London what should I wear?"
content = chat(user_question, tools=[get_current_weather])
print("Content of the response:", content)

