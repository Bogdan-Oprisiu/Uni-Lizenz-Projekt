# bpe_tokenizer.py

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
                        "training_files must be provided if tokenizer file is missing and train_if_missing is True.")
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
        special_tokens = [
            "[PAD]", "[UNK]", "[SOS]", "[EOS]", "[SEP]",
            "{", "}", "[", "]", ":", ",",
            "\"commandLanguage\"", "\"errors\"", "\"commands\"", "\"parameters\"",
            "\"name\"", "\"description\"", "\"distance\"", "\"acceleration\"",
            "\"angle\"", "\"direction\""
        ]

        # 5. Create a trainer with a vocabulary size and special tokens.
        trainer = trainers.BpeTrainer(vocab_size=10000, special_tokens=special_tokens)

        # Check if the files exist and print a warning if any are missing.
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
