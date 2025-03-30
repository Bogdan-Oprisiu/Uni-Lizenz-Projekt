import json

import torch
import torch.quantization
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers.utils import move_cache

##############################################################################
# 1. Optional: Clear or migrate HF model cache
##############################################################################
move_cache()

##############################################################################
# 2. Load the abstract command structure from your JSON definition
##############################################################################
with open("../possible_commands.json", "r", encoding="utf-8") as f:
    possible_commands_str = f.read()

##############################################################################
# 3. Initialize tokenizer & T5 model, then apply dynamic quantization
##############################################################################
model_name = "t5-small"
tokenizer = T5TokenizerFast.from_pretrained(model_name, model_max_length=512)

model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()  # Set model to evaluation mode

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)


##############################################################################
# 4. Helper: top-k extraction + float16 quantization of probabilities
##############################################################################
def topk_float16(prob_tensor: torch.Tensor, k=5):
    """
    Extract the top-k probabilities from `prob_tensor` and quantize them
    from 32-bit float to 16-bit float. Return a list of (token_id, prob16) pairs.
    prob_tensor shape: [vocab_size]
    """
    # 1. Select top-k
    top_values, top_indices = torch.topk(prob_tensor, k)
    # 2. Renormalize so they sum to 1 (since softmax can have numerical variance)
    top_values = top_values / top_values.sum()
    # 3. Convert to float16
    top_values_16 = top_values.half()  # Float16 quantization
    # 4. Build (token_id, quantized_prob) pairs
    pairs = list(zip(top_indices.tolist(), top_values_16.tolist()))
    return pairs


##############################################################################
# 5. Teacher generation function
#    - Builds a prompt from your "possible_commands" instructions
#    - Generates with the quantized T5 model
#    - Collects top-5 float16 probabilities for each decoder step
##############################################################################
def get_teacher_probs(input_text, temperature=2.0, max_new_tokens=50):
    """
    Constructs a prompt with the abstract command structure, tokenizes it,
    and then generates output using the quantized T5 model.

    Returns:
      step_probs_top5 (list): For each generated token, a list of top-5
        (token_id, float16_probability) pairs.
    """
    # Prepare the prompt
    prompt = (
        f"Using the following command structure:\n{possible_commands_str}\n\n"
        f"Translate command to json: {input_text}"
    )
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    # Generate (use no_grad to reduce overhead)
    with torch.no_grad():
        outputs = quantized_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False  # greedy or beam search
        )

    # For each step's logits, convert to probabilities & keep top-5 float16
    step_probs_top5 = []
    for step_score in outputs.scores:  # each is [batch_size=1, vocab_size]
        probs = torch.softmax(step_score / temperature, dim=-1)
        top5_16 = topk_float16(probs[0], k=5)
        step_probs_top5.append(top5_16)

    return step_probs_top5


##############################################################################
# 6. Load the basic unlabeled data from a text file & process each command
##############################################################################
with open("../training_data/synthetic_unlabeled_robot_commands.txt", "r", encoding="utf-8") as f:
    # Each line is one command; filter out any empty lines.
    commands = [line.strip() for line in f if line.strip()]

results = []

for idx, input_text in enumerate(commands):
    print(f"Processing command {idx + 1} of {len(commands)}: {input_text}")

    try:
        # Generate and get top-5 probabilities for each decoding step
        probs_stepwise = get_teacher_probs(input_text, temperature=2.0)
    except RuntimeError as e:
        print(f"Runtime error at command index {idx}: {e}")
        break

    # Store only the probability distributions (matching the position of the command)
    results.append({
        "input_index": idx,  # so we know which line in the text file
        "soft_targets_top5_float16": probs_stepwise
    })

    # Optionally free GPU cache
    torch.cuda.empty_cache()

##############################################################################
# 7. Write the final results (with top-5 float16 probabilities) to JSON
##############################################################################
with open("teacher_outputs.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Teacher outputs (with top-5 float16 probabilities) have been saved to teacher_outputs_basic.json")
