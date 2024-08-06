from prompete import Chat

from dataclasses import dataclass
from typing import List
import json

# Define a simple function to be called by the LLM
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather in a given location"""
    # In a real scenario, you would call an actual weather API here
    weather_data = {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "forecast": ["sunny", "windy"]
    }
    return json.dumps(weather_data)

# Create a Chat instance
chat = Chat(model="gpt-4o-mini")

# Define the user's question
user_question = "What's the weather like in London?"
content = chat(user_question, tools=[get_current_weather])

# Process the response
outputs = chat.process()

# Print the results
print("User:", user_question)
print("Content of the response:", content)
print("Weather data:", outputs[0] if outputs else "No weather data retrieved")
