# autocast just makes things worse!
# don't even bother using it.
from time import time

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)

input_size = 1
output_size = 1
hidden_size = 512
num_data = 100000

# seeds
seed = 1234
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
np.random.seed(seed)

# hyper-parameters
num_epochs = 5
learning_rate = 0.01

# toy dataset
x_train = np.random.rand(num_data, input_size)
y_train = np.cos(2 * np.pi * x_train) + 0.1 * np.random.randn(num_data, input_size)

# regression model
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.GELU(),
    nn.Linear(hidden_size, hidden_size),
    nn.GELU(),
    nn.Linear(hidden_size, output_size),
)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)

# train the model
x_train = torch.from_numpy(x_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))

global_start_time = time()
for epoch in range(num_epochs):
    start_time = time()

    # forward pass
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, cache_enabled=True):
        outputs = model(x_train)
        loss = criterion(outputs, y_train)

    # backward and optimize
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: Loss: {loss.item():.6e} in {time()-start_time:.1f}s.")
print(f"training done in {time()-global_start_time:.1f}s.")
