import torch.cuda
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn, optim

from model import LeNet

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

train_data = datasets.CIFAR10(root="../Dataset/CIFAR10", train=True, transform=transform, download=False)  # 训练50000张
test_data = datasets.CIFAR10(root="../Dataset/CIFAR10", train=False, transform=transform, download=False)  # 测试10000张
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers= 0)
test_loader = DataLoader(test_data, batch_size=10000, shuffle=False, num_workers=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = LeNet().to(device=device)
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.005)
test_iter = iter(test_loader)
test_imgs, test_labels = test_iter.next()
test_imgs = test_imgs.to(device)
test_labels = test_labels.to(device)
epoch = 20
print(device)
print("------训练开始-----")
for i in range(epoch):
    tot_loss = 0.0
    for data in train_loader:
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = net(imgs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_loss = tot_loss + loss.item()
    with torch.no_grad():
        outputs = net(test_imgs)
        predict_y = torch.max(outputs, dim=1)[1]
        accurcy = (predict_y == test_labels).sum().item() / test_labels.size(0)
        print("epoch= %d, loss = %.3lf, accury= %.3lf" %(i+1, tot_loss, accurcy))
        tot_loss = 0.0
print("-----训练结束-----")
torch.save(net.state_dict(), "./train_result/LeNet.pth")