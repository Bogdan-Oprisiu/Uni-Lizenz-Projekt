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


#####################################
# Dummy Model Architecture
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
# Helper Function to Process Teacher Steps
#####################################

def process_teacher_steps(teacher_steps_list, batch_seq_len, vocab_size):
    """
    Convert teacher_steps (a list per sample of a list per timestep of top-k [token_id, probability] pairs)
    into a tensor of shape (batch, batch_seq_len, vocab_size).
    For each timestep, only the top-k indices are filled (others remain 0).
    Optionally, the row is normalized so that it sums to 1.
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
            if teacher_tensor[t].sum() > 0:
                teacher_tensor[t] = teacher_tensor[t] / teacher_tensor[t].sum()
        batch_teacher_tensor.append(teacher_tensor)
    return torch.stack(batch_teacher_tensor, dim=0)  # (batch, batch_seq_len, vocab_size)


#####################################
# Training Function
#####################################

def train(tokenizer, labeled_json_path, distilled_json_path, batch_size=4, epochs=3, device="cpu"):
    # Assume tokenizer has a property 'vocab_size'
    vocab_size = tokenizer.vocab_size
    model = SimpleSeq2Seq(vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Loss functions:
    # CrossEntropyLoss for hard labels (ignoring pad token id 0)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    # KLDivLoss for teacher soft targets; expects log-probs as input.
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    # Create datasets & dataloaders
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
        collate_fn=collate_fn_labeled
    )
    distilled_loader = DataLoader(
        distilled_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_distilled
    )

    # For plotting losses
    labeled_losses = []
    distilled_losses = []
    combined_losses = []

    model.train()
    for epoch in range(epochs):
        for (batch_labeled, batch_distilled) in zip(labeled_loader, distilled_loader):
            # ---- Process Hard-labeled batch ----
            input_ids_labeled = batch_labeled["input_ids"].to(device)  # shape: (B, seq_len)
            target_ids = batch_labeled["target_ids"].to(device)  # shape: (B, seq_len)

            logits_labeled = model(input_ids_labeled)  # (B, seq_len, vocab_size)
            # Flatten and compute cross entropy loss
            loss_labeled = ce_loss_fn(logits_labeled.view(-1, vocab_size),
                                      target_ids.view(-1))

            # ---- Process Teacher-distilled batch ----
            input_ids_distilled = batch_distilled["input_ids"].to(device)  # (B, seq_len)
            logits_distilled = model(input_ids_distilled)  # (B, seq_len, vocab_size)
            log_probs = torch.log_softmax(logits_distilled, dim=-1)  # log-probs for KLDivLoss

            # Process teacher steps: convert raw teacher_steps into a tensor of shape (B, seq_len, vocab_size)
            batch_seq_len = input_ids_distilled.size(1)
            teacher_dist = process_teacher_steps(batch_distilled["teacher_steps"],
                                                 batch_seq_len,
                                                 vocab_size).to(device)
            loss_distilled = kl_loss_fn(log_probs, teacher_dist)

            # ---- Combine Losses ----
            loss = loss_labeled + loss_distilled

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record losses for plotting
            labeled_losses.append(loss_labeled.item())
            distilled_losses.append(loss_distilled.item())
            combined_losses.append(loss.item())

            print(f"Epoch {epoch + 1} | Labeled Loss: {loss_labeled.item():.4f} | "
                  f"Distilled Loss: {loss_distilled.item():.4f} | Combined Loss: {loss.item():.4f}")

    # Plot loss curves using matplotlib
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


#####################################
# Main: Setup Tokenizer & Run Training
#####################################

if __name__ == "__main__":
    # Dummy tokenizer for demonstration.
    # Replace this with your own tokenizer that implements .encode() and has a .vocab_size property.
    class DummyTokenizer:
        def __init__(self):
            self.vocab = {"<pad>": 0, "move": 1, "left": 2, "right": 3, "forward": 4, "back": 5}
            self.vocab_size = 1000  # Assume a vocabulary size of 1000 tokens

        def encode(self, text):
            # For demo, split text on spaces and map to token ids; unknown tokens get id 6.
            return [self.vocab.get(token, 6) for token in text.split()]


    tokenizer = DummyTokenizer()

    # Paths to your JSON files (update these paths as needed)
    labeled_json_path = "..\\training_data\\synthetic_basic_labeled_robot_commands.json"
    distilled_json_path = "..\\training_data\\synthetic_basic_unlabeled_robot_commands.txt"

    # Run training
    train(tokenizer, labeled_json_path, distilled_json_path,
          batch_size=4, epochs=10, device="cuda")
