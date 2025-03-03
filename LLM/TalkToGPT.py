from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Load a pre-trained model and tokenizer
model_name = "gpt2"  # or "gpt2-medium", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Create a text-generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Interactive chat loop
print("Chat with the AI (type 'quit' to exit):")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # Create a prompt (you can add context or conversation history here)
    prompt = f"User: {user_input}\nAI:"
    response = generator(prompt, max_length=100, num_return_sequences=1)
    ai_output = response[0]["generated_text"]

    # Post-process to remove the prompt part if needed
    print("AI:", ai_output.split("AI:")[-1].strip())
