import logging
from prompete import Chat, ToolList
from pprint import pprint
from typing import Callable

# Configure logging
prompete_logger = logging.getLogger("answerbot.chat")
#prompete_logger.setLevel(logging.DEBUG)

logger = logging.getLogger("thinking_loop")
logger.setLevel(logging.DEBUG)
#logging.basicConfig(
#    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
#)

class ThoughtOrganizer:
    def __init__(self):
        self.thoughts = []

    def add_thought(self, thought: str) -> str:
        """Add a thought to the problem"""
        self.thoughts.append(thought)
        record = f"Thought number {len(self.thoughts)} added:\n\n{thought}"
        logger.debug(record)
        return record

    def clear_thoughts(self) -> str:
        """Clear all thoughts"""
        self.thoughts = []
        return "Thoughts cleared."
    
    def get_tools(self) -> list[Callable]:
        return [self.add_thought]


MAX_THOUGHTS = 20
# Create a Chat instance
chat = Chat(model="gpt-4o", max_loops = MAX_THOUGHTS - 1, tool_manager=ThoughtOrganizer())


problem = """7 axles are equally spaced around a circle. A gear is placed on each axle such
that each gear is engaged with the gear to its left and the gear to its right. The gears are
numbered 1 to 7 around the circle. If gear 3 were rotated clockwise, in which direction would
gear 7 rotate?"""


# Define the user's question
user_question = f"""Please analyze the following problem:

{problem}
You can use the `add_thought` function to collect your thoughts on the problem.
Think step by step - don't rush, at each step add just one thought.
This might be a tricky question - please check the consistency of your thinking.
After {MAX_THOUGHTS} thoughts you need to formulate your answer.
"""
answer = chat(user_question)

# Print the results
print("User: ", user_question)
print("Answer: ", answer)

pprint(chat.messages)

pprint(chat.tool_manager.thoughts)