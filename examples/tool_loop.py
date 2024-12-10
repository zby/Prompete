import logging
from prompete import Chat, ToolList
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather in a given location"""
    # In a real scenario, you would call an actual weather API here
    return {
        "location": location,
        "temperature": 28,
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }

# Create a Chat instance
chat = Chat(model="gpt-4o-mini", tool_manager=ToolList([get_current_weather]))

# Define the user's question  
user_question = "Please check the weather in London (using the `get_current_weather` function) and suggest an appropriate outfit."
answer = chat(user_question)

# Print the results
print("User: ", user_question)
print("Answer: ", answer)

pprint(chat.messages)
