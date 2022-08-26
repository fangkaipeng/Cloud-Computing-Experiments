import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = LeNet().to(device=device)
net.load_state_dict(torch.load("./train_result/LeNet.pth"))
im = Image.open("./predict_img/cat2.jfif")
im = transform(im)
im = im.to(device)
im = torch.unsqueeze(im, dim=0)

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1]
print(classes[int(predict)])
