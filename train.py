import torch
import torch.nn as nn
from dataloader import AudioDataset
from model import AudioClassifier
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# reproducability
torch.manual_seed(42)

# Define the directories
noise_dir = "data/noise2"
clap_dir = "data/claps"

# Create the dataset
dataset = AudioDataset(noise_dir, clap_dir)

# Define the size of the splits
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the model
model = AudioClassifier()

# Move the model to the GPU
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.02)

num_epochs = 5

# Train the model
for epoch in range(num_epochs):
    model.train()

    train_losses = []
    correct_train_predictions = 0
    total_train_predictions = 0

    for i, (inputs, labels) in enumerate(train_dataloader):
        # Move the data to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)
        train_losses.append(loss.item())

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train_predictions += labels.size(0)
        correct_train_predictions += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()

    train_loss = sum(train_losses) / len(train_losses)
    train_accuracy = correct_train_predictions / total_train_predictions

    print(
        f"Epoch {epoch + 1}: Train loss: {train_loss}, Train Accuracy: {train_accuracy}"
    )

    model.eval()

    # Validation Phase
    with torch.no_grad():
        val_losses = []
        correct_predictions = 0
        total_predictions = 0
        for i, (inputs, labels) in enumerate(val_dataloader):
            # Move the data to the GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        val_loss = sum(val_losses) / len(val_losses)
        val_accuracy = correct_predictions / total_predictions

    print(
        f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}"
    )
# Save the model
torch.save(model.state_dict(), "audio_classifier.pth")
