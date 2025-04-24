import os
import tempfile

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import Lowercase, Replace, Sequence


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

        # 1.1. Set a normalizer to:
        #      - convert text to lowercase, and
        #      - remove non-ASCII characters.
        tokenizer.normalizer = Sequence([
            Lowercase(),
            Replace(pattern=r"[^\x00-\x7F]+", content=""),
        ])

        # 2. Set a pre-tokenizer that splits text into initial tokens.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        # 3. Set a decoder to convert tokens back into text.
        tokenizer.decoder = decoders.ByteLevel()

        # 4. Define special tokens for JSON punctuation and domain-specific keys.
        special_tokens = [
            "[PAD]", "[UNK]", "[SOS]", "[EOS]", "[SEP]",
            "{", "}", "[", "]", ":", ",",
            "\"commandLanguage\"", "\"errors\"", "\"commands\"", "\"parameters\"",
            "\"name\"", "\"description\"", "\"distance\"", "\"acceleration\"",
            "\"angle\"", "\"direction\"",
            "\"INVALID_COMMAND\"", "\"MISSING_PARAMETER\"", "\"INVALID_PARAMETER_TYPE\"",
            "\"OUT_OF_RANGE\"", "\"SYNTAX_ERROR\"", "\"UNSUPPORTED_UNIT\""
        ]

        trainer = trainers.BpeTrainer(vocab_size=10000, special_tokens=special_tokens)

        # 5. Check if files exist
        for f in files:
            if not os.path.exists(f):
                print(f"File not found: {f}")

        # 6. Train the tokenizer on the given files.
        tokenizer.train(files, trainer)

        # 7. Configure a post-processor to add [SOS], [EOS], [SEP].
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[SOS] $A [EOS]",
            pair="[SOS] $A [EOS] [SEP] $B [EOS]",
            special_tokens=[
                ("[SOS]", tokenizer.token_to_id("[SOS]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )

        # 8. Save the tokenizer
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer training complete and saved to {tokenizer_path}")


if __name__ == "__main__":
    # Original dictionary file and other training files
    english_dict_path = "../../training_data/tokenizer/google-10000-english-no-swears.txt"
    other_files = [
        "..\\training_data\\basic_data\\synthetic_basic_labeled_robot_commands_json.txt",
        "..\\training_data\\multiple_parameter_data\\synthetic_labeled_robot_commands_with_accel_json.txt"
    ]

    # other_files = []

    # ------------------------
    # 1. Read the dictionary file and replicate lines in memory.
    # ------------------------
    dict_lines = []
    replicate_factor = 5  # Increase this value if you want more weight
    if os.path.exists(english_dict_path):
        with open(english_dict_path, "r", encoding="utf-8") as f:
            original_dict_lines = f.read().splitlines()
        dict_lines = original_dict_lines * replicate_factor
    else:
        print(f"Dictionary file not found: {english_dict_path}")

    # ------------------------
    # 2. Write the replicated lines to a temporary file
    # ------------------------
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_dict.txt") as tmp_file:
        tmp_dict_path = tmp_file.name
        for line in dict_lines:
            tmp_file.write(line + "\n")

    # Now we have a new temp file that includes the dictionary with increased weight
    # appended multiple times.

    # ------------------------
    # 3. Combine the dictionary temp file with your other files
    # ------------------------
    training_files = [tmp_dict_path] + other_files

    # ------------------------
    # 4. Train the tokenizer
    # ------------------------
    tokenizer_wrapper = HFTokenizerWrapper(
        tokenizer_path="bpe_tokenizer.json",
        train_if_missing=True,
        training_files=training_files
    )

    # Cleanup: remove tmp_dict_path if you prefer
    os.remove(tmp_dict_path)
