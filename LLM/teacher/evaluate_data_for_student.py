import json
import os
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers.utils import move_cache
import torch.quantization

# Clear or migrate the cache (optional)
move_cache()

# Load the abstract command structure from the JSON file
with open("../training_data/possible_commands.json", "r", encoding="utf-8") as f:
    possible_commands_str = f.read()

# Use a smaller model (e.g., T5-small)
model_name = "t5-small"
tokenizer = T5TokenizerFast.from_pretrained(model_name, model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()  # Set model to evaluation mode

# Apply dynamic quantization to reduce memory usage
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)


def get_teacher_outputs(input_text, temperature=2.0, max_new_tokens=100):
    """
    Constructs a prompt with the abstract command structure, tokenizes it,
    and then generates output using the quantized teacher model.

    Returns:
      generated_text: the generated output text (e.g., JSON command)
      soft_targets: a list of soft target probability distributions for each generated token
    """
    prompt = (
        f"Using the following command structure:\n{possible_commands_str}\n\n"
        f"Translate command to json: {input_text}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    outputs = quantized_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        do_sample=False
    )
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    soft_targets = [torch.softmax(score / temperature, dim=-1).tolist() for score in outputs.scores]
    return generated_text, soft_targets


# Load training data (JSON array of command objects)
with open("../training_data/synthetic_robot_commands.json", "r", encoding="utf-8") as f:
    commands = json.load(f)

results = []
checkpoint_file = "teacher_outputs_checkpoint.json"
checkpoint_interval = 10
start_index = 0

if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    start_index = len(results)
    print(f"Resuming from checkpoint. Starting at command index {start_index}")

for idx, command in enumerate(commands[start_index:], start=start_index):
    input_text = command["input_text"]
    expected_output = command["expected_output"]
    print("Processing command:", input_text)
    try:
        generated_text, soft_targets = get_teacher_outputs(input_text, temperature=2.0)
    except RuntimeError as e:
        print(f"Runtime error at command index {idx}: {e}")
        break
    result = {
        "input_text": input_text,
        "expected_output": expected_output,
        "generated_text": generated_text,
        "soft_targets": soft_targets
    }
    results.append(result)
    if (idx + 1) % checkpoint_interval == 0:
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Checkpoint saved at command index {idx + 1}")
    torch.cuda.empty_cache()

with open("teacher_outputs.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Teacher outputs have been saved to teacher_outputs.json")
