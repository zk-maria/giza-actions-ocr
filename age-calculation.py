import torch
import torch.nn as nn
import torch.onnx
from datetime import datetime

# Define a simple neural network model
class AgeCalculatorModel(nn.Module):
    def __init__(self):
        super(AgeCalculatorModel, self).__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)

# Create the model instance
model = AgeCalculatorModel()

# Define dummy inputs and outputs for training
# Here we are generating synthetic data to train the model
# The input is a tensor containing (birth year, birth month, birth day)
# The output is a tensor containing the age

# Function to generate synthetic data
def generate_data(num_samples):
    inputs = []
    outputs = []
    current_year = datetime.today().year
    current_month = datetime.today().month
    current_day = datetime.today().day

    for _ in range(num_samples):
        birth_year = torch.randint(1950, 2020, (1,)).item()
        birth_month = torch.randint(1, 13, (1,)).item()
        birth_day = torch.randint(1, 29, (1,)).item()
        age = current_year - birth_year - ((current_month, current_day) < (birth_month, birth_day))
        inputs.append([birth_year, birth_month, birth_day])
        outputs.append([age])
    
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)

# Generate synthetic data
inputs, outputs = generate_data(1000)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs_pred = model(inputs)
    loss = criterion(outputs_pred, outputs)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Export the trained model to ONNX
dummy_input = torch.tensor([[2000, 1, 1]], dtype=torch.float32)
torch.onnx.export(model, dummy_input, "age_calculator_model.onnx", 
                  input_names=['birth_date'], 
                  output_names=['age'],
                  dynamic_axes={'birth_date': {0: 'batch_size'}, 'age': {0: 'batch_size'}})

print("Model exported to age_calculator_model.onnx")
