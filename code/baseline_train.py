import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import wandb
from tqdm import tqdm

# Hyperparameters & Setup
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 3e-4 # Standard AdamW LR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WandB
wandb.init(
    project="cifar10-100epoch",
    name="Baseline-AdamW",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "AdamW",
        "model": "ResNet18"
    }
)

# Data Pipeline CIFAR-10
print("Setting up CIFAR-10")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Model Setup - ResNet-18
print("Initializing ResNet-18")
# weights=None, train from scratch
model = models.resnet18(weights=None) 
# CIFAR-10 has 10 classes, ResNet18 defaults to 1000, replace the final layer
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(DEVICE)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

scheduler_adamw = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
schedulers = [scheduler_adamw]

# Training Loop
print(f"Starting Training on {DEVICE}")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    # Wrap trainloader in tqdm for a progress bar
    pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_train_loss = running_loss / len(trainloader)

    # Validation Loop
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(testloader)
    val_accuracy = 100 * correct / total

    print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
    
    # Log to WandB
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_accuracy": val_accuracy
    })

    # Scheduler step
    scheduler_adamw.step()

wandb.finish()
print("Baseline training complete")