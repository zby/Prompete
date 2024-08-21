from prompete import Chat

from typing import Annotated


def get_current_weather(
    location: str, unit: Annotated[str, "celsius or fahrenheit"]
) -> str:
    # in strict mode there cannot be default values in the function signature
    # this is why we use Annotated to provide a description of the unit parameter
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
content = chat(user_question, tools=[get_current_weather], strict=True)

# Process the response
outputs = chat.process()

# Print the results
print("User:", user_question)
print("Content of the response:", content)
print("Weather data:", outputs[0] if outputs else "No weather data retrieved")
