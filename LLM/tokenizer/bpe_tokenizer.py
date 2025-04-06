"""
bpe_tokenizer.py

Below are some ideas and best practices to consider for improving your tokenizer and
making it more likely for an LLM to reliably generate JSON of the desired structure.

----------------------------------------------------------------
1. Add More Domain-Specific Special Tokens
----------------------------------------------------------------
    - For keys like "commandLanguage", "errors", "parameters", "distance", "acceleration", etc.
    - For punctuation used in JSON: '{', '}', '[', ']', ':', ','.
    - This prevents the BPE merge steps from splitting them incorrectly and helps
      preserve the exact JSON structure in generation.

----------------------------------------------------------------
2. Include a Richer JSON “Skeleton” in the Training Data
----------------------------------------------------------------
    - Provide full, well-formed JSON examples to your tokenizer during training.
    - Repeated usage of braces, quotes, colons, commas, and typical JSON fields helps the
      tokenizer treat these as significant tokens or subwords.

----------------------------------------------------------------
3. Experiment with Vocab Size
----------------------------------------------------------------
    - 10k tokens is a reasonable starting point, but verify if it properly handles
      domain terms (e.g., “acceleration” remains one token or a small set of subwords).
    - Adjust up or down depending on your domain’s complexity.

----------------------------------------------------------------
4. Consider a Custom Normalizer or Post-Processor
----------------------------------------------------------------
    - Tokenizer post-processing can add [SOS], [EOS], [SEP], etc.
    - Potentially auto-insert certain JSON punctuation if desired.
    - Be cautious not to over-constrain or hamper the model’s ability to adapt.

----------------------------------------------------------------
5. Add Additional Special Tokens for JSON-Level Control
----------------------------------------------------------------
    - If you have repeated patterns or blocks, consider using tokens like <CMD>...</CMD>.
    - Train the model to understand these tags, though this is optional/advanced.

----------------------------------------------------------------
6. Don’t Forget Coverage of User Inputs
----------------------------------------------------------------
    - Make sure your tokenizer sees examples of user inputs (e.g., “move forward 100cm”).
    - The model can then better parse and respond with the correct JSON schema.

----------------------------------------------------------------
7. Validate the JSON Output Post-Generation
----------------------------------------------------------------
    - Even with perfect tokenization, LLMs can generate invalid JSON.
    - Use a JSON parser or schema validator on the output, catch errors,
      and respond with appropriate error codes if needed.
"""

import os

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


class HFTokenizerWrapper:
    """
    A wrapper for the HuggingFace Tokenizers library's BPE tokenizer.
    This class provides an interface with an `encode` method and a `vocab_size` attribute.

    If the tokenizer file does not exist and `train_if_missing` is True,
    it will train the tokenizer using the provided training files.
    """

    def __init__(self, tokenizer_path="bpe_tokenizer.json", train_if_missing=False, training_files=None):
        if not os.path.exists(tokenizer_path):
            if train_if_missing:
                if training_files is None:
                    raise ValueError(
                        "training_files must be provided if tokenizer file is missing "
                        "and train_if_missing is True."
                    )
                self.train_tokenizer(tokenizer_path, training_files)
            else:
                raise FileNotFoundError(f"Tokenizer file {tokenizer_path} not found.")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text):
        """Encodes a given text string into a list of token IDs."""
        return self.tokenizer.encode(text).ids

    def train_tokenizer(self, tokenizer_path, files):
        """
        Train a BPE tokenizer on the given list of files and save it to tokenizer_path.
        """
        # 1. Initialize a BPE model.
        tokenizer = Tokenizer(models.BPE())

        # 2. Set a pre-tokenizer that splits text into initial tokens.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        # 3. Set a decoder to convert tokens back into text.
        tokenizer.decoder = decoders.ByteLevel()

        # 4. Define special tokens (including punctuation and domain-specific tokens).
        #    Here, you might consider adding even more domain-specific tokens or JSON structure tokens.
        special_tokens = [
            "[PAD]", "[UNK]", "[SOS]", "[EOS]", "[SEP]",
            "{", "}", "[", "]", ":", ",",
            "\"commandLanguage\"", "\"errors\"", "\"commands\"", "\"parameters\"",
            "\"name\"", "\"description\"", "\"distance\"", "\"acceleration\"",
            "\"angle\"", "\"direction\"",
            # Add more here if needed...
        ]

        # 5. Create a trainer with a vocabulary size and special tokens.
        trainer = trainers.BpeTrainer(vocab_size=10000, special_tokens=special_tokens)

        # 6. Check if the files exist and print a warning if any are missing.
        for f in files:
            if not os.path.exists(f):
                print(f"File not found: {f}")

        # 7. Train the tokenizer on the given files.
        tokenizer.train(files, trainer)

        # 8. Configure a post-processor to automatically add start, end, and separator tokens.
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[SOS] $A [EOS]",
            pair="[SOS] $A [EOS] [SEP] $B [EOS]",
            special_tokens=[
                ("[SOS]", tokenizer.token_to_id("[SOS]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )

        # 9. Save the tokenizer to a file for later use.
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer training complete and saved to {tokenizer_path}")


# If you want to allow training from the command line, you can add:
if __name__ == "__main__":
    # Define the list of files used for training your tokenizer.
    training_files = [
        "..\\possible_commands.json",
        "..\\training_data\\synthetic_unlabeled_robot_commands.txt",
        "..\\training_data\\synthetic_basic_unlabeled_robot_commands.txt"
    ]

    # Train the tokenizer and save to bpe_tokenizer.json.
    HFTokenizerWrapper(tokenizer_path="bpe_tokenizer.json", train_if_missing=True, training_files=training_files)
