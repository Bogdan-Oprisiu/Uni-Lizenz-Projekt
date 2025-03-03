import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Hyperparameters
# -------------------------------
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
HIDDEN_DIM = 128  # size of the hidden layer

# -------------------------------
# 2. Load Data from train.csv
# -------------------------------
# Your CSV now has these columns:
# ['text_id', 'full_text', 'cohesion', 'syntax', 'vocabulary',
#  'phraseology', 'grammar', 'conventions']
df = pd.read_csv("train.csv")
# Uncomment the next line to see the column names:
# print(df.columns)

# -------------------------------
# 3. Train/Test Split
# -------------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Separate features and labels.
# Here we assume "full_text" is the text input and "cohesion" is the target label.
X_train_text = train_df["full_text"].values
y_train = train_df["cohesion"].values
X_test_text = test_df["full_text"].values
y_test = test_df["cohesion"].values

# -------------------------------
# 4. TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

# Fit on the training text only
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)


# -------------------------------
# 5. Create Torch Datasets
# -------------------------------
class EssayDataset(Dataset):
    def __init__(self, X, y):
        # X is a sparse matrix from TfidfVectorizer
        self.X = torch.tensor(X.toarray(), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = EssayDataset(X_train_tfidf, y_train)
test_dataset = EssayDataset(X_test_tfidf, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -------------------------------
# 6. Define a Two-Layer Network
# -------------------------------
class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Determine number of classes based on unique labels in the training set.
num_classes = len(set(y_train))  # Adjust as needed for your task

model = TwoLayerNet(input_dim=X_train_tfidf.shape[1], hidden_dim=HIDDEN_DIM, output_dim=num_classes)

# -------------------------------
# 7. Define Loss and Optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# -------------------------------
# 8. Training & Evaluation Functions
# -------------------------------
def train_one_epoch(loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_x.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_x.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# -------------------------------
# 9. Main Training Loop
# -------------------------------
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(train_loader)
    test_loss, test_acc = evaluate(test_loader)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
