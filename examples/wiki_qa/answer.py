from prompete.chat import Chat, SystemPrompt
from prompete.ReplayLLM import ReplayLiteLLM
from wiki_tool import WikipediaTool
from pydantic import BaseModel, Field
from pathlib import Path

MODEL = "gpt-4o-mini"
#MODEL = "anthropic/claude-3-5-haiku-latest"
MAX_LLM_REQUESTS = 5

SAVE_DIR = Path(__file__).parent / "save"


# The question we want to answer
QUESTION = "What was the first successful powered aircraft?"

# QUESTION = "What was the first major battle in the Ukrainian War?"
# QUESTION = "What were the main publications by the Nobel Prize winner in economics in 2023?"
# QUESTION = "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
# QUESTION = 'Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?'
# QUESTION = "how old was Donald Tusk when he died?"
# QUESTION = "how many keys does a US-ANSI keyboard have on it?"
# QUESTION = "How many children does Donald Tusk have?"
# QUESTION = "The director of the romantic comedy \"Big Stone Gap\" is based in what New York city?"
# QUESTION = "When Poland became elective monarchy?"
# QUESTION = "Were Scott Derrickson and Ed Wood of the same nationality?"
# QUESTION = "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?"
# QUESTION = "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?"
# QUESTION = "What is the weight proportion of oxygen in water?"
# QUESTION = "Czy dane kardy kredytowej są danymi osobowymi w Polsce"
# QUESTION = "How much is two plus two"
# QUESTION = "Who is older, Annie Morton or Terry Richardson?"

# QUESTION = "What are the concrete steps proposed to ensure AI safety?"
# QUESTION = 'What are the steps required to authorize the training of generative AI?'

# QUESTION = "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?"
QUESTION = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
#QUESTION = "The arena where the Lewiston Maineiacs played their home games can seat how many people?"
# QUESTION = "What is the seating capacity of Androscoggin Bank Colisée?"
#QUESTION = "Who portrayed Corliss Archer in the film Kiss and Tell?"
#QUESTION = "When did Wordsworth initially attack Burke?"


def record_reflection(
    what_have_we_learned: str = Field(..., description="Summary of what information relevant to the user question have we discovered so far."),
    comment: str = Field(..., description="A general comment on the retrieved information."),
    relevant_quotes: list[str] = Field(..., description="A list of relevant literal quotes from the retrieved information since last reflection."),
    new_sources: list[str] = Field(..., description="A list of new urls mentioned in the retrieved information that should be checked later."),
) -> str:
    print(f"What have we learned: {what_have_we_learned}")
    print(f"Comment: {comment}")
    print(f"Relevant quotes: {relevant_quotes}")
    print(f"New sources: {new_sources}")
    return "Note recorded"


def main():
    # Initialize the Wikipedia tool
    wiki_tool = WikipediaTool()
    # Get all the available tools from WikipediaTool
    tools = wiki_tool.get_all_tools()
    #tools.append(record_reflection)
    
    llm_provider = ReplayLiteLLM(replay_dir=SAVE_DIR, replay_count=1)
     
    # Create a chat instance with GPT-4
    chat = Chat(
        model=MODEL,
        llm_provider=llm_provider,
        system_prompt="""You are a helpful AI assistant that answers questions using Wikipedia.
You are precise and double check your work.
You have access to various Wikipedia tools to help you find and read information.
You should never access the tools in parallel - always use one tool at a time.
If you're not sure about something, say so.""",
        tools=tools,
    )
    
    response = None
    
    for i in range (2):
        
        if i == 0:
            message = f"The user question is: {QUESTION}, you can use the wikipedia tools for research. When you have enough information to answer the question, please do so."
        else:
            message = f"Just for focuse - the user question is: {QUESTION}, proceed with your research untill you can formulate an answer."
   
        response = chat(
            message,
            max_llm_requests=MAX_LLM_REQUESTS
        )
        if response:
            break
        
        #chat.append("Please review the work so far")
        #chat.complete_once( tool_choice={"type": "function", "function": {"name": "record_reflection"}})
        reflection = chat(
            """Summarize the research so far.
            Make a list of findings and hypotheses that might help to answer the user question with their supporting evidence.
            The supporting evidence for a finding should always contain an exact quote and the source url, hypotheses don't require quotes.
            Never modify the quotes, don't use elipsis or other modifications the quotes should be exact matches.
            Be critical - note assumptions that have been made and analyse if they need correction.
            Be comprehensive. Revise previous reflections if needed.""",
            tool_choice="none"
        )
        
        print("Reflection:\n", reflection)
        print()

    print()
    print("\nAnswer:", response)
 
if __name__ == "__main__":
    main()
