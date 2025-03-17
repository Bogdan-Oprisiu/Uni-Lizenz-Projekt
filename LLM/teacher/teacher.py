import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers.utils import move_cache

move_cache()

# Load the abstract command structure from the JSON file
with open("../training_data/possible_commands.json", "r") as f:
    possible_commands_str = f.read()

# Set the teacher model name and load T5-Large along with its tokenizer.
# Using the Fast tokenizer can help avoid legacy issues.
model_name = "t5-large"
tokenizer = T5TokenizerFast.from_pretrained(model_name, model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode


def get_teacher_outputs(input_text, temperature=2.0, max_new_tokens=100):
    """
    Constructs a prompt including the abstract command structure, tokenizes the input (with truncation),
    and returns the teacher model's generated sequence along with soft target distributions (logits converted to probabilities)
    for each generated token.
    """
    # Construct the prompt with the abstract command structure.
    prompt = (
        f"Using the following command structure:\n{possible_commands_str}\n\n"
        f"Translate command to json: {input_text}"
    )

    # Tokenize the input prompt with a maximum length and enable truncation to avoid length errors.
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    # Generate output using the teacher model.
    # We use output_scores and return_dict_in_generate to get the logits at each generation step.
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        do_sample=False  # Use greedy decoding
    )

    # outputs.sequences contains the generated tokens.
    # outputs.scores is a tuple with one tensor per generated token, each of shapes (batch_size, vocab_size).
    # We convert each score tensor to soft target probabilities using softmax with temperature scaling.
    soft_targets_list = []
    for score in outputs.scores:
        soft = torch.softmax(score / temperature, dim=-1)
        soft_targets_list.append(soft)

    # Stack along the time dimension: resulting shape will be (batch_size, max_new_tokens, vocab_size)
    soft_targets = torch.stack(soft_targets_list, dim=1)

    return inputs, outputs.sequences, soft_targets


if __name__ == "__main__":
    # Example input command.
    sample_input = "Move forward 100 cm with acceleration 10."

    # Obtain teacher outputs.
    inputs, generated_tokens, soft_targets = get_teacher_outputs(sample_input, temperature=2.0)

    # Print details about the results.
    print("Input Tokens (IDs):", inputs.input_ids)
    tokenized_text = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    print("Tokenized Text:", tokenized_text)
    print("Generated Sequence (IDs):", generated_tokens)
    gen_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print("Generated Text:", gen_text)
    print("Soft Targets Shape:", soft_targets.shape)

    # Display soft target probabilities for the first token of the generated sequence (first 10 values)
    first_token_soft = soft_targets[0, 0, :10]
    print("Soft Targets (first token, first 10 values):", first_token_soft)
