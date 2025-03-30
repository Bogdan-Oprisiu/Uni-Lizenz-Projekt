import json

import torch
from torch.utils.data import Dataset


class LabeledCommandsDataset(Dataset):
    """
    Hard-labeled dataset that loads JSON of the form:
      [
        {
          "input_text": "move left 264.00 cm.",
          "expected_output": {
            "action": "left",
            "parameters": {
              "distance": 264
            }
          }
        },
        ...
      ]
    """

    def __init__(self, json_path, tokenizer, max_length=128):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        input_text = record["input_text"]
        # Convert the JSON object to a string for training
        # e.g. {"action": "left", "parameters": {"distance": 264}}
        # You might choose a different serialization strategy.
        target_text = str(record["expected_output"])

        input_ids = self.tokenizer.encode(input_text)[: self.max_length]
        target_ids = self.tokenizer.encode(target_text)[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long)
        }


class DistilledCommandsDataset(Dataset):
    """
    Teacher-distilled dataset that loads JSON of the form:
      [
        {
          "input_index": 0,
          "generated_text": "...some text from teacher...",
          "soft_targets_top5_float16": [
              [[token_id, prob], ...],  # step 0
              [[token_id, prob], ...],  # step 1
              ...
          ]
        },
        ...
      ]

    We will:
      - Use 'generated_text' as the model's input (or possibly the original text if present).
      - Store the teacher's top-k distributions for each timestep in `soft_targets_top5_float16`.
    """

    def __init__(self, json_path, tokenizer, max_length=128):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        # The teacher’s "generated_text" or possibly "input_text" if available
        input_text = record.get("input_text", record.get("generated_text", ""))
        input_ids = self.tokenizer.encode(input_text)[: self.max_length]

        # The teacher’s distribution data
        teacher_steps = record["soft_targets_top5_float16"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "teacher_steps": teacher_steps  # raw list of lists (we'll handle it in the collate function)
        }


def collate_fn_labeled(batch):
    """
    Collate for the LabeledCommandsDataset:
      - Pad input_ids and target_ids to the maximum length in the batch (across both).
    """
    input_ids_list = [item["input_ids"] for item in batch]
    target_ids_list = [item["target_ids"] for item in batch]

    # Use a common max length across both inputs and targets
    max_len = max(
        max(len(x) for x in input_ids_list),
        max(len(x) for x in target_ids_list)
    )

    padded_inputs = []
    padded_targets = []
    for inp, tgt in zip(input_ids_list, target_ids_list):
        # Pad with 0s (assuming 0 is your pad token id)
        inp_pad = torch.cat([
            inp,
            torch.zeros(max_len - len(inp), dtype=torch.long)
        ])
        tgt_pad = torch.cat([
            tgt,
            torch.zeros(max_len - len(tgt), dtype=torch.long)
        ])
        padded_inputs.append(inp_pad)
        padded_targets.append(tgt_pad)

    return {
        "input_ids": torch.stack(padded_inputs, dim=0),  # (batch, max_len)
        "target_ids": torch.stack(padded_targets, dim=0)  # (batch, max_len)
    }


def collate_fn_distilled(batch):
    """
    Collate for the DistilledCommandsDataset:
      - Pad input_ids to the max length in the batch.
      - Keep teacher_steps as-is (list of lists), for custom processing in your loss function.
    """
    input_ids_list = [item["input_ids"] for item in batch]
    max_inp_len = max(len(x) for x in input_ids_list)

    padded_inputs = []
    teacher_steps_list = []
    for inp_dict in batch:
        inp = inp_dict["input_ids"]
        teacher_steps = inp_dict["teacher_steps"]
        inp_pad = torch.cat([
            inp,
            torch.zeros(max_inp_len - len(inp), dtype=torch.long)
        ])
        padded_inputs.append(inp_pad)
        teacher_steps_list.append(teacher_steps)

    return {
        "input_ids": torch.stack(padded_inputs, dim=0),
        "teacher_steps": teacher_steps_list
    }


class DistilledCommandsDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Read the file line by line.
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Encode the text and create a dummy teacher_steps structure.
        text = self.data[idx]
        # Wrap the token ids with torch.tensor to convert the list into a tensor.
        input_ids = torch.tensor(self.tokenizer.encode(text)[: self.max_length], dtype=torch.long)
        # Create a dummy teacher_steps: a list (per timestep) of one [token, probability] pair.
        teacher_steps = [[[token, 1.0]] for token in input_ids.tolist()]
        return {"input_ids": input_ids, "teacher_steps": teacher_steps}
