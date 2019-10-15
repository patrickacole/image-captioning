import torch
import torchvision
import torchvision.transforms as transforms

from torch import nn
from tqdm import tqdm


class LargeNet(nn.Module):
    def __init__(self):
        super(LargeNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 1200),
            nn.ReLU(),
            nn.Linear(1200, 1200),
            nn.ReLU(),
            nn.Linear(1200, 1200),
            nn.ReLU(),
            nn.Linear(1200, 10)
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.main(x)
        return x


def iterate_dataloader(dataloader, net, optimizer, train):
    criterion = torch.nn.CrossEntropyLoss()
    pbar = tqdm(enumerate(dataloader, 1), desc='Train' if train else 'False')
    for i, (x, y) in pbar:
        if train:
            optimizer.zero_grad()

        out = net(x)
        loss = criterion(out, y)
        pred_labels = torch.argmax(out, dim=1)

        if train:
            loss.backward()
            optimizer.step()

        acc = torch.mean((pred_labels == y).type(torch.FloatTensor))
        pbar.set_postfix_str(f"{acc * 100:.1f}% {loss.item():.3f}L")


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    net = LargeNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    iterate_dataloader(trainloader, net, optimizer, train=True)
    iterate_dataloader(testloader, net, optimizer, train=False)
