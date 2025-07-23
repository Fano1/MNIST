import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import matplotlib.pyplot as plt
from PIL import Image

#device testing and hyper parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 10
n_epoch = 100
lr = 0.001
inputSize = 28*28
hiddenSize = 500
classesNum = 10

#data and data loader

train_data = dataset.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_data = dataset.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)

#data loader
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

for images, labels in train_loader:
    print(f"Image batch shape: {images.shape}")   # [B, 1, 28, 28]
    print(f"Label batch shape: {labels.shape}")   # [B]
    print(f"First label in batch: {labels[0]}")
    
    # Optional: display the first image
    import matplotlib.pyplot as plt
    plt.imshow(images[0].squeeze(), cmap="gray")
    plt.title(f"Label: {labels[0].item()}")
    plt.axis("off")
    plt.show()
    break  # remove this to go through the whole dataset

#layer 

class NeuralNetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, classesNum):
        super().__init__()
        self.layer1 = nn.Linear(inputSize, hiddenSize)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hiddenSize, classesNum)

    def forward(self, X):
        out = self.layer1(X)
        out = self.relu(out)
        out = self.layer2(out)
        return out

model = NeuralNetwork(inputSize, hiddenSize, classesNum).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
citration = nn.CrossEntropyLoss()

#training loop
for epoch in range(5):
    for index, (img, lable) in enumerate(train_loader):
        image = img.reshape(-1, 28*28).to(device)
        lable = lable.to(device)

        prediction = model(image)
        lossVal = citration(prediction, lable)
        lossVal.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch [{epoch+1}/{5}], Step [{index+1}], Loss: {lossVal.item():.4f}")

def check_accuracy(loader, model):
    correct = 0
    total = 0
    model.eval()  # eval mode disables dropout/batchnorm if present

    with torch.no_grad():
        for images, labels in loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            outputs = model(images)                # (B, num_classes)
            _, predicted = torch.max(outputs, 1)   # get class indices
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

check_accuracy(test_loader, model)

# Load your image
img_path = "test.png"  # or use test_data[0][0] to grab from dataset
image = Image.open(img_path).convert('L')  # convert to grayscale

# Transform it (same as in training)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),  # scales to [0,1] and gives (1,28,28)
])

image = transform(image)

# Reshape and send to device
image = image.view(-1, 28*28).to(device)  # shape: (1, 784)

# Make sure model is in eval mode
model.eval()

with torch.no_grad():
    output = model(image)  # shape: (1, num_classes)
    _, predicted = torch.max(output, 1)

print(f"Predicted digit: {predicted.item()}")
torch.save(model.state_dict(), 'model_params.pth')
