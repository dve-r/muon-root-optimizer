import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import wandb
from tqdm import tqdm

class Muon(Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-Schulz.
    Specifically designed for 2D hidden layer parameters (Conv2D, Linear).
    """
    def __init__(self, params, lr=0.02, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                assert grad.ndim >= 2, "Muon strictly requires 2D+ parameters."
                
                state = self.state[p]
                
                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                buf = state['momentum_buffer']
                
                # Update momentum (standard SGD momentum)
                buf.mul_(momentum).add_(grad)
                
                # Reshape 4D convolutions to 2D matrices for orthogonalization
                original_shape = buf.shape
                if buf.ndim > 2:
                    G = buf.view(original_shape[0], -1)
                else:
                    G = buf
                
                # Speed Optimization: Transpose tall matrices to save compute
                transposed = False
                if G.shape[0] > G.shape[1]:
                    G = G.T
                    transposed = True
                
                # Normalize matrix
                X = G / (G.norm(p='fro') + 1e-8)
                
                # 5-Step Newton Schulz
                a, b, c = 3.4445, -4.7750, 2.0315
                for _ in range(5):
                    A = X @ X.T
                    B = A @ X
                    C = A @ B
                    X = a * X + b * B + c * C
                
                # Undo transpose if applied
                if transposed:
                    X = X.T
                
                # Scale back up and reshape to original dimensions
                dim_out, dim_in = G.shape[0] if not transposed else G.shape[1], G.shape[1] if not transposed else G.shape[0]
                scale = max(1, dim_out / dim_in) ** 0.5
                update = (X * scale).view(original_shape)
                
                # Apply update
                p.add_(update, alpha=-lr)

# Hyperparameters & Setup
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 3e-4 # Standard AdamW LR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WandB
wandb.init(
    project="cifar10-100epoch",
    name="Muon-Experiment",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Muon+AdamW",
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
print("Routing parameters for Muon + AdamW")

muon_params = []
adamw_params = []

for name, param in model.named_parameters():
    # If the parameter has 2 or more dimensions (like Conv2d weights or Linear weights)
    if param.ndim >= 2:
        muon_params.append(param)
    # If the parameter is 1D (BatchNorm weights, biases)
    else:
        adamw_params.append(param)

# Initialize the dual optimizers
# Note: Muon typically requires a slightly larger learning rate than AdamW
optimizer_muon = Muon(muon_params, lr=0.01, momentum=0.95)
optimizer_adamw = optim.AdamW(adamw_params, lr=3e-4, weight_decay=0.01)

# Group them into a list so we can step both easily in the training loop
optimizers = [optimizer_muon, optimizer_adamw]

# CosineAnnealingLR scheduler
scheduler_muon = optim.lr_scheduler.CosineAnnealingLR(optimizer_muon, T_max=EPOCHS)
scheduler_adamw = optim.lr_scheduler.CosineAnnealingLR(optimizer_adamw, T_max=EPOCHS)
schedulers = [scheduler_muon, scheduler_adamw]

# Training Loop
print(f"Starting Training on {DEVICE}")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    # Wrap trainloader in tqdm for a progress bar
    pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Zero gradients
        for opt in optimizers:
            opt.zero_grad()
            
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Step both optimizers
        for opt in optimizers:
            opt.step()

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
    for sched in schedulers:
        sched.step()

wandb.finish()
print("Muon training complete")