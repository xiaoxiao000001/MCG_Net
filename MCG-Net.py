import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Multi-Scale Self-Attention Module (MSSA)
class MSSA(nn.Module):
    def __init__(self, in_channels, seq_len, num_scales=3, reduction=16):
        super(MSSA, self).__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_scales = num_scales

        # Multi-scale convolution kernels
        kernel_sizes = [1, 3, 5]  # Kernel sizes for different scales
        paddings = [0, 1, 2]      # Padding to maintain consistent feature map size

        # Create multi-scale branches
        self.scale_branches = nn.ModuleList()
        for i in range(num_scales):
            branch = nn.Sequential(
                nn.Conv1d(in_channels, in_channels // reduction,
                          kernel_size=kernel_sizes[i], padding=paddings[i], bias=False),
                nn.BatchNorm1d(in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
            self.scale_branches.append(branch)

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(in_channels * num_scales, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Self-attention mechanism
        self.self_attn = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [Batch, Channels, Time]
        b, c, t = x.size()

        # Multi-scale feature extraction
        scale_features = []
        for branch in self.scale_branches:
            scale_features.append(branch(x) * x)  # Apply attention weights

        # Feature fusion
        fused_features = torch.cat(scale_features, dim=1)  # [B, C*num_scales, T]
        multi_scale_output = self.fusion(fused_features)   # [B, C, T]

        # Apply self-attention
        attn_weights = self.self_attn(multi_scale_output)
        output = multi_scale_output * attn_weights

        return output


# Model using MSSA
class CNNGRU(nn.Module):
    def __init__(self, num_classes=5, seq_len=15):  # Assume sequence length is 15
        super().__init__()
        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool1d(kernel_size=1),
            nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool1d(kernel_size=1),
            nn.Conv1d(64, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool1d(kernel_size=1)
        )

        # Replace MCA with MSSA
        self.mssa = MSSA(in_channels=128, seq_len=seq_len)
        self.flatten = nn.Flatten()

        # GRU layer for sequential modeling
        self.gru = nn.GRU(input_size=1, hidden_size=64, num_layers=1,
                          batch_first=True, bidirectional=False)

        self.gru_fc = nn.Linear(64, 64)
        self.combined_fc = nn.Linear(128 * seq_len + 64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # CNN feature extraction
        # Reshape input from [B, T, 1] to [B, C, T] for Conv1d
        cnn_features = self.conv_layers(x.permute(0, 2, 1))

        # Apply MSSA
        attended_features = self.mssa(cnn_features)

        # Flatten CNN features with dropout
        cnn_flatten = self.dropout(attended_features.flatten(1))  # [B, C*T]

        # GRU branch processing
        gru_input = x.permute(0, 1, 2)  # [B, T, 1]
        gru_output, _ = self.gru(gru_input)
        # Take the last time step output
        gru_feat = self.gru_fc(gru_output[:, -1, :])

        # Feature concatenation and final classification
        combined = torch.cat((cnn_flatten, gru_feat), dim=1)
        return self.combined_fc(combined)


# Data loading and preprocessing
# Load dataset from Excel file
data = pd.read_excel('PCC_SBS+VTI.xlsx')
# Extract labels (first column) and features (remaining columns)
labels = data.iloc[:, 0].values
reflectance = data.iloc[:, 1:].values
seq_len = reflectance.shape[1]  # Get actual sequence length

# Split dataset into train, validation and test sets
X_train, X_test, y_train, y_test = train_test_split(reflectance, labels, test_size=0.3, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=True)

# Convert to PyTorch tensors and adjust dimensions to [B, T, 1]
X_train = torch.Tensor(X_train).float().unsqueeze(-1).to(device)
y_train = torch.from_numpy(y_train).long().to(device)
X_val = torch.Tensor(X_val).float().unsqueeze(-1).to(device)
y_val = torch.from_numpy(y_val).long().to(device)
X_test = torch.Tensor(X_test).float().unsqueeze(-1).to(device)
y_test = torch.from_numpy(y_test).long().to(device)

# Initialize model with actual sequence length
model = CNNGRU(num_classes=5, seq_len=seq_len).to(device)
# Set up optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.CrossEntropyLoss()

# Create DataLoaders for batch processing
train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Training process
epochs = 300
# Initialize history trackers for loss and accuracy
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []
test_loss_history = []  # New: test loss history
test_acc_history = []   # New: test accuracy history

for epoch in range(epochs):
    # Training phase
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == y_batch).sum().item()
        total_samples += y_batch.size(0)

    # Calculate average training loss and accuracy
    train_loss = total_loss / total_samples
    train_acc = total_correct / total_samples
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * x_batch.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.size(0)

    # Calculate average validation loss and accuracy
    val_loss = val_loss / len(val_data)
    val_acc = val_correct / val_total
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    # Test phase (new)
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * x_batch.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == y_batch).sum().item()
            test_total += y_batch.size(0)

    # Calculate average test loss and accuracy
    test_loss = test_loss / len(test_data)
    test_acc = test_correct / test_total
    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# Model evaluation and visualization
model.eval()
with torch.no_grad():
    # Get predictions for test set
    y_pred = torch.argmax(model(X_test), dim=1).cpu().numpy()
y_test_np = y_test.cpu().numpy()

# Print classification report
print("\nClassification Report:")
report_dict = classification_report(y_test_np, y_pred, target_names=[str(i) for i in range(5)], output_dict=True)

# Format classification report for better readability
headers = ["precision", "recall", "f1-score", "support"]
rows = [headers]
for label in sorted(report_dict.keys()):
    if label == "accuracy":
        row = [label, "", "", f"{report_dict[label]:.4f}", f"{report_dict['macro avg']['support']:.6f}"]
        rows.append(row)
    elif label in ["macro avg", "weighted avg"]:
        row = [label]
        for metric in headers[:-1]:
            value = report_dict[label][metric]
            row.append(f"{value:.4f}")
        row.append(f"{report_dict[label]['support']:.6f}")
        rows.append(row)
    else:
        row = [label]
        for metric in headers[:-1]:
            value = report_dict[label][metric]
            row.append(f"{value:.4f}")
        row.append(f"{report_dict[label]['support']}")
        rows.append(row)

# Print formatted classification report
col_width = max(len(header) for header in headers) + 2
for row in rows:
    print("".join(str(cell).ljust(col_width) for cell in row))

# Calculate and print confusion matrix
confusion_mat = confusion_matrix(y_test_np, y_pred)
confusion_mat = confusion_mat.astype(np.float64)
confusion_mat = np.round(confusion_mat, 4)
print("Confusion Matrix:\n", confusion_mat)

# Calculate overall accuracy and kappa coefficient
oa = 100.0 * np.sum(y_test_np == y_pred) / len(y_test_np)
kappa = cohen_kappa_score(y_test_np, y_pred)
print(f"Overall Accuracy (OA): {oa:.4f}%")
print(f"Kappa Coefficient: {kappa:.4f}")

# Plot loss curves (updated: include test loss)
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.plot(test_loss_history, label='Test Loss')  # New: test loss curve
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(ls='--')
plt.show()

# Plot accuracy curves (updated: include test accuracy)
plt.figure(figsize=(10, 5))
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.plot(test_acc_history, label='Test Accuracy')  # New: test accuracy curve
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(ls='--')
plt.show()

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt=".4f", cmap="YlGn",
            xticklabels=[str(i) for i in range(5)], yticklabels=[str(i) for i in range(5)])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
