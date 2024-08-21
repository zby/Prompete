from prompete import Chat

# Note: This example assumes that you have set up your API credentials
# in your environment variables. Depending on the model you choose, you'll need to set:
# - OPENAI_API_KEY for OpenAI models
# - ANTHROPIC_API_KEY for Anthropic models
# - GROQ_API_KEY for Groq models
# For example:
# export OPENAI_API_KEY='your-openai-api-key-here'
# export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
# export GROQ_API_KEY='your-groq-api-key-here'

# Uncomment the desired model:
# model = "claude-3-haiku-20240307"  # Anthropic model
# model = "llama2-70b-4096"  # Groq model
model = "gpt-4o-mini"  # OpenAI model

# Create a Chat instance with a system prompt
chat = Chat(
    model=model,
    system_prompt="You are a helpful assistant specializing in Python programming.",
)

# Start the conversation
user_message = "What's the difference between a list and a tuple in Python?"
response = chat(user_message)

print("User:", user_message)
print("AI:", response)

# Continue the conversation
user_message = "Can you give me an example of when to use a tuple instead of a list?"
response = chat(user_message)

print("\nUser:", user_message)
print("AI:", response)

# Access the chat history
print("\nChat History:")
for message in chat.messages:
    print(f"{message['role'].capitalize()}: {message['content'][:50]}...")
