import logging
from prompete import Chat
from pprint import pprint
from typing import Callable

# Configure logging
prompete_logger = logging.getLogger("prompete.chat")
#prompete_logger.setLevel(logging.DEBUG)

logger = logging.getLogger("thinking_loop")
#logger.setLevel(logging.DEBUG)
#logging.basicConfig(
#    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
#)

MAX_THOUGHTS = 7
MODEL = "gpt-4o-mini"
#MODEL = "anthropic/claude-3-5-sonnet-latest"
MODEL = "anthropic/claude-3-5-haiku-latest"


class ThoughtOrganizer:
    def __init__(self):
        self.thoughts = []
        self.summary_frequency = 5

    def add_thought(self, thought: str) -> str:
        """Add a thought to the problem"""
        self.thoughts.append(thought)
        record = f"Thought number {len(self.thoughts)} added:\n\n{thought}"
        logger.debug(record)
        return record

    def summarize(self, summary: str) -> str:
        """Summarize the thoughts so far and check consistency between them"""
        self.thoughts.append(summary)
        record = f"Thought number {len(self.thoughts)} added:\n\n{summary}"
        logger.debug(record)
        return record

    def clear_thoughts(self) -> str:
        """Clear all thoughts"""
        self.thoughts = []
        return "Thoughts cleared."


problem = """7 axles are equally spaced around a circle. A gear is placed on each axle such
that each gear is engaged with the gear to its left and the gear to its right. The gears are
numbered 1 to 7 around the circle. If gear 3 were rotated clockwise, in which direction would
gear 7 rotate?"""


# Define the user's question
user_question = f"""Please analyze the following problem:

{problem}
You can use the `add_thought` function to collect your thoughts on the problem.
At every fifth step, you should summarize your thoughts and check for consistency.
Think step by step - don't rush, at each step add just one thought.
This might be a tricky question - please check the consistency of your thinking
and also check if you analyzed all the conditions set in the problem statement.

After {MAX_THOUGHTS} thoughts you need to write a report.
It should contain a reasoning path leading to a conclusion - the proposed answer.
"""

thought_organizer = ThoughtOrganizer()
chat = Chat(model=MODEL, tools=[thought_organizer.add_thought])

answer = chat(user_question, max_llm_requests=MAX_THOUGHTS)
print(f"Answer: {answer}")
#summary = chat("Summarize your thoughts and check for consistency.", tool_choice="none")
#print(f"Summary: {summary}")

# Print the results
#print("User: ", user_question)
#print("Answer: ", answer)

#pprint(chat.messages)

#pprint(chat.tool_manager.thoughts)

 