from prompete import Chat


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather in a given location"""
    # In a real scenario, you would call an actual weather API here
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }


# Create a Chat instance
chat = Chat(model="gpt-4o-mini")

# Define the user's question
user_question = "What's the weather like in London?"
content = chat(user_question, tools=[get_current_weather])

# Print the results
print("User:", user_question)
print("Content of the response:", content)
