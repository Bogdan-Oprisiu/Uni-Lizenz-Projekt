from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# 1. Initialize a BPE model
tokenizer = Tokenizer(models.BPE())

# 2. Set a pre-tokenizer that splits text into initial tokens.
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

# 3. Set a decoder to convert tokens back into text.
tokenizer.decoder = decoders.ByteLevel()

# 4. Define special tokens, now including [SEP]
special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[SEP]", "{", "}", "[", "]", ":", ","]

# 5. Create a trainer with your desired vocabulary size and special tokens.
trainer = trainers.BpeTrainer(vocab_size=10000, special_tokens=special_tokens)

# 6. Provide your training data (list of files).
files = ["..\\training_data\\synthetic_robot_commands.json"]

# 7. Train the tokenizer
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
tokenizer.save("bpe_tokenizer.json")

print("Tokenizer training complete and saved to bpe_tokenizer.json")
