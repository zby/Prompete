import logging
from pprint import pprint

from prompete.chat import Chat


logger = logging.getLogger("prompete.chat")
#logger.setLevel(logging.DEBUG)


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

# Example using complete_once
print("\nExample using complete_once:")
chat = Chat(model="gpt-3.5-turbo")
chat.append("What is 5 + 7?")

response, outputs = chat.complete_once(tools=[add_numbers])

print("Response:", response.content)
if outputs:
    print("Tool outputs:", outputs)
else:
    print("No tools were called")

# Example using __call__
print("\nExample using __call__:")
chat2 = Chat(model="gpt-3.5-turbo", tools=[add_numbers])

result = chat2("What is 8 + 3?")
print("Final response:", result)
#pprint(chat2.messages_to_dict_list())