import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

# Initialize model, input, loss function, and optimizer
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)
targets = torch.randn(5, 1000)  # Random target tensor matching the output size of ResNet
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Profile the full training loop
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for epoch in range(5):  # Profile 5 epochs
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        optimizer.step()

        prof.step()  # Mark the end of each epoch

# Print profiling results sorted by CPU time
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


