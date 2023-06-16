from sklearn.datasets import load_wine
from sklearn import metrics, model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# Load the wine dataset
dataset = load_wine()
X = dataset.data
y = dataset.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert the NumPy arrays to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
# Define the hyperparameters
input_size = 13
num_classes = 3
num_epochs = 1000
batch_size = 13
learning_rate = 0.001

# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.hidden1 = nn.Linear(input_size, 15)
        self.linear = nn.Linear(15, num_classes)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.linear(x)
        return x

# Create an instance of the model and move it to the device
model = LogisticRegression(input_size, num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

import matplotlib.pyplot as plt

# Initialize empty lists to store the training and validation loss
train_loss = []
valid_loss = []

# Train the model
for epoch in range(num_epochs):
    # Shuffle the training data
    perm = torch.randperm(len(y_train))
    X_train_shuffled = X_train[perm]
    y_train_shuffled = y_train[perm]

    # Loop over batches of data
    for i in range(0, len(y_train), batch_size):
        # Get the batch of data
        xb = X_train_shuffled[i:i+batch_size].to(device)
        labels = y_train_shuffled[i:i+batch_size].to(device)

        # Forward pass
        outputs = model(xb)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate the model on the validation set
    with torch.no_grad():
        outputs = model(X_test.to(device))
        loss = criterion(outputs, y_test.to(device))
        valid_loss.append(loss.item())
        
    # Record the training loss
    train_loss.append(loss.item())

    # Print the loss
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs} Training Loss: {train_loss[-1]:.4f} Validation Loss: {valid_loss[-1]:.4f}')
    
    with torch.no_grad():
        outputs = model(X_train.to(device))
        _, predicted = torch.max(outputs, 1)
        train_acc = metrics.accuracy_score(y_train, predicted.cpu().numpy())
        if (epoch+1) % 100 == 0:
            print(f'Training accuracy after epoch {epoch+1}: {train_acc:.4f}')
        
        
# Plot the training and validation loss
plt.plot(train_loss, label='Training Loss')
plt.plot(valid_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Evaluate the model on the test set
with torch.no_grad():
    outputs = model(X_test.to(device))
    _, predicted = torch.max(outputs, 1)
    y_pred = predicted.cpu().numpy()

acc = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc:.4f}')