import json
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers.utils import move_cache

# Clear or migrate the cache (optional)
move_cache()

# Load the abstract command structure from the JSON file
with open("../training_data/possible_commands.json", "r", encoding="utf-8") as f:
    possible_commands_str = f.read()

# Set the teacher model name and load T5-Large along with its fast tokenizer.
# We set model_max_length=512 to handle long inputs.
model_name = "t5-large"
tokenizer = T5TokenizerFast.from_pretrained(model_name, model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()  # Set model to evaluation mode


def get_teacher_outputs(input_text, temperature=2.0, max_new_tokens=100):
    """
    Constructs a prompt with the abstract command structure, tokenizes it,
    and then generates output using the teacher model.

    Returns:
      generated_text: the generated output text (e.g., JSON command)
      soft_targets: a list of soft target probability distributions for each generated token
    """
    # Build the prompt by prepending the command structure
    prompt = (
        f"Using the following command structure:\n{possible_commands_str}\n\n"
        f"Translate command to json: {input_text}"
    )

    # Tokenize the prompt with truncation (max_length=512)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    # Generate output and return generation scores
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        do_sample=False  # Greedy decoding; adjust if needed
    )

    # Convert generated tokens to text
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Process the scores: each element in outputs.scores is a tensor of shape (batch_size, vocab_size)
    # Apply softmax with temperature scaling to each and convert them to lists.
    soft_targets = [torch.softmax(score / temperature, dim=-1).tolist() for score in outputs.scores]

    return generated_text, soft_targets


# Load your training data: a file containing a JSON array of command objects.
with open("../training_data/synthetic_robot_commands.json", "r", encoding="utf-8") as f:
    commands = json.load(f)

results = []
for command in commands:
    input_text = command["input_text"]
    expected_output = command["expected_output"]
    print("Processing command:", input_text)

    # Get the teacher's generated output and soft targets for the given command.
    generated_text, soft_targets = get_teacher_outputs(input_text, temperature=2.0)

    # Store everything in a dictionary for later use in student training.
    result = {
        "input_text": input_text,
        "expected_output": expected_output,
        "generated_text": generated_text,
        "soft_targets": soft_targets
    }
    results.append(result)

# Save the collected teacher outputs for use in student training.
with open("teacher_outputs.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Teacher outputs have been saved to teacher_outputs.json")
