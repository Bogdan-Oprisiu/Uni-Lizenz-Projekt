import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import (
    LabeledCommandsDataset,
    DistilledCommandsDataset,
    collate_fn_labeled,
    collate_fn_distilled
)
from V1.tokenizer.bpe_tokenizer import HFTokenizerWrapper


# If your custom tokenizer is in a separate file/module, import it here:
# from my_tokenizer import MyTokenizer


#####################################
# Simple Seq2Seq Model Architecture
#####################################

class SimpleSeq2Seq(nn.Module):
    """
    A simple sequence-to-sequence model that:
      - Embeds input tokens,
      - Passes them through a Transformer encoder,
      - Projects to the vocabulary.
    """

    def __init__(self, vocab_size, embed_dim=64, num_layers=2, num_heads=4, dropout=0.1):
        super(SimpleSeq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        emb = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        emb = emb.transpose(0, 1)  # (seq_len, batch, embed_dim) for transformer
        encoded = self.encoder(emb)  # (seq_len, batch, embed_dim)
        encoded = encoded.transpose(0, 1)  # (batch, seq_len, embed_dim)
        logits = self.fc(encoded)  # (batch, seq_len, vocab_size)
        return logits


#####################################
# Helper Function for Teacher Steps
#####################################

def process_teacher_steps(teacher_steps_list, batch_seq_len, vocab_size):
    """
    Convert teacher_steps (a list per sample of a list per timestep of top-k [token_id, probability] pairs)
    into a tensor of shape (batch, batch_seq_len, vocab_size).
    For each timestep, only the top-k indices are filled (others remain 0).
    We'll also normalize each timestep's distribution to sum to 1 (if non-zero).
    """
    batch_teacher_tensor = []
    for teacher_steps in teacher_steps_list:
        L = len(teacher_steps)
        # Initialize a (batch_seq_len x vocab_size) tensor with zeros
        teacher_tensor = torch.zeros(batch_seq_len, vocab_size)
        for t in range(min(L, batch_seq_len)):
            # teacher_steps[t] is a list of [token_id, probability] pairs.
            for pair in teacher_steps[t]:
                token_id, prob = pair
                token_id = int(token_id)
                teacher_tensor[t, token_id] = prob
            # Normalize if there's any probability > 0
            row_sum = teacher_tensor[t].sum()
            if row_sum > 0:
                teacher_tensor[t] /= row_sum
        batch_teacher_tensor.append(teacher_tensor)
    return torch.stack(batch_teacher_tensor, dim=0)  # (batch, batch_seq_len, vocab_size)


#####################################
# Training Function
#####################################

def train(tokenizer, labeled_json_path, distilled_json_path, batch_size=4, epochs=3, device="cpu"):
    # 1) Build the model & optimizer
    vocab_size = tokenizer.vocab_size
    model = SimpleSeq2Seq(vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 2) Define losses
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # For hard labels
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")  # For teacher soft targets (KL)

    # 3) Create Datasets & DataLoaders
    labeled_dataset = LabeledCommandsDataset(
        json_path=labeled_json_path,
        tokenizer=tokenizer,
        max_length=128
    )
    distilled_dataset = DistilledCommandsDataset(
        json_path=distilled_json_path,
        tokenizer=tokenizer,
        max_length=128
    )

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_labeled  # pads input & target
    )
    distilled_loader = DataLoader(
        distilled_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_distilled  # pads input, leaves teacher steps as list
    )

    # 4) Lists to store losses for plotting
    labeled_losses = []
    distilled_losses = []
    combined_losses = []

    # 5) Training Loop
    model.train()
    for epoch in range(epochs):
        # zip stops when the shorter loader ends => each epoch will use min(#batches_labeled, #batches_distilled)
        for (batch_labeled, batch_distilled) in zip(labeled_loader, distilled_loader):
            # (a) Hard-labeled batch
            input_ids_labeled = batch_labeled["input_ids"].to(device)  # (B, seq_len_inp)
            target_ids = batch_labeled["target_ids"].to(device)  # (B, seq_len_tgt)

            # Forward pass
            logits_labeled = model(input_ids_labeled)  # (B, seq_len_inp, vocab_size)

            # Flatten for CE => shapes must match, so we rely on collate_fn_labeled
            loss_labeled = ce_loss_fn(
                logits_labeled.view(-1, vocab_size),  # (B*seq_len_inp, vocab_size)
                target_ids.view(-1)  # (B*seq_len_inp,)
            )

            # (b) Teacher-distilled batch
            input_ids_distilled = batch_distilled["input_ids"].to(device)  # (B, seq_len)
            logits_distilled = model(input_ids_distilled)  # (B, seq_len, vocab_size)
            log_probs = torch.log_softmax(logits_distilled, dim=-1)  # for KL

            # Convert teacher steps => shape (B, seq_len, vocab_size)
            batch_seq_len = input_ids_distilled.size(1)
            teacher_dist = process_teacher_steps(batch_distilled["teacher_steps"], batch_seq_len, vocab_size).to(device)

            # KLDivLoss
            loss_distilled = kl_loss_fn(log_probs, teacher_dist)

            # (c) Combine losses
            loss = loss_labeled + loss_distilled

            # (d) Backprop & update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # (e) Record losses
            labeled_losses.append(loss_labeled.item())
            distilled_losses.append(loss_distilled.item())
            combined_losses.append(loss.item())

        print(f"[Epoch {epoch + 1}] "
              f"Labeled Loss: {loss_labeled.item():.4f} | "
              f"Distilled Loss: {loss_distilled.item():.4f} | "
              f"Combined Loss: {loss.item():.4f}")

    # 6) Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(labeled_losses, label="Labeled Loss (CE)")
    plt.plot(distilled_losses, label="Distilled Loss (KL)")
    plt.plot(combined_losses, label="Combined Loss")
    plt.xlabel("Batch Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.show()

    print("Training complete!")

    # Return the trained model so we can save it outside
    return model


#####################################
# Main: Load Your Custom BPE Tokenizer & Run Training
#####################################

if __name__ == "__main__":
    # Instantiate my custom BPE tokenizer from the JSON file.
    tokenizer_path = "../tokenizer/bpe_tokenizer.json"
    tokenizer = HFTokenizerWrapper(tokenizer_path)

    # File paths to your datasets (update these paths as needed)
    labeled_json_path = "../training_data/synthetic_basic_labeled_robot_commands.json"
    distilled_json_path = "../../training_data/basic_data/synthetic_basic_unlabeled_robot_commands.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train(
        tokenizer=tokenizer,
        labeled_json_path=labeled_json_path,
        distilled_json_path=distilled_json_path,
        batch_size=8,
        epochs=20,
        device=device
    )

    # Save the model's state dictionary for later reuse
    torch.save(model.state_dict(), "trained_basic_model.pt")
    print("Model saved to trained_model.pt")
