import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        # Decoder
        self.conv5 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        x1 = nn.functional.relu(self.conv1(x))
        x2 = nn.functional.relu(self.conv2(nn.functional.max_pool2d(x1, 2)))
        x3 = nn.functional.relu(self.conv3(nn.functional.max_pool2d(x2, 2)))
        x4 = nn.functional.relu(self.conv4(nn.functional.max_pool2d(x3, 2)))
        # Decoder
        x5 = nn.functional.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = nn.functional.relu(self.conv5(torch.cat([x3, x5], dim=1)))
        x6 = nn.functional.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = nn.functional.relu(self.conv6(torch.cat([x2, x6], dim=1)))
        x7 = nn.functional.interpolate(x6, scale_factor=2, mode='bilinear', align_corners=False)
        x7 = nn.functional.relu(self.conv7(torch.cat([x1, x7], dim=1)))
        x8 = nn.functional.sigmoid(self.conv8(x7))
        return x8

# Define the training function
def train_unet(data_folder, num_epochs=10, batch_size=8, learning_rate=1e-4):
    # Load the dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(data_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Define the U-Net model
    model = UNet()
    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, running_loss/(i+1)))
    # Save the trained model