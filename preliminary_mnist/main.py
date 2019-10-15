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


class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 10)
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.main(x)
        return x


def iterate_dataloader(dataloader, net, optimizer, train, softtarget=None, temp=3):
    hard_criterion = torch.nn.CrossEntropyLoss()
    soft_criterion = torch.nn.BCEWithLogitsLoss()

    pbar = tqdm(enumerate(dataloader, 1),
                desc=f"{'Train' if train else 'Test'} {'Soft' if softtarget is not None else 'Hard'}")
    for i, (x, y) in pbar:
        soft_y = None
        if softtarget is not None:
            with torch.no_grad():
                soft_y = softtarget(x)
                soft_y = soft_y ** temp
                soft_y = torch.nn.functional.softmax(soft_y, dim=1)

        if train:
            optimizer.zero_grad()

        out = net(x)
        hard_loss = hard_criterion(out, y)
        soft_loss = torch.zeros(1)
        if softtarget is not None:
            soft_loss = soft_criterion(out, soft_y)

        loss = hard_loss + soft_loss

        pred_labels = torch.argmax(out, dim=1)

        if train:
            loss.backward()
            optimizer.step()

        acc = torch.mean((pred_labels == y).type(torch.FloatTensor))
        pbar.set_postfix_str(f"{acc * 100:.1f}% {hard_loss.item():.3f}HL {soft_loss.item():.3f}SL")


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    large_net = LargeNet()
    large_opt = torch.optim.Adam(large_net.parameters(), lr=1e-3)
    iterate_dataloader(trainloader, large_net, large_opt, train=True)
    iterate_dataloader(testloader, large_net, large_opt, train=False)

    small_net = SmallNet()
    small_opt = torch.optim.Adam(small_net.parameters(), lr=1e-3)
    iterate_dataloader(trainloader, small_net, small_opt, train=True)
    iterate_dataloader(testloader, small_net, small_opt, train=False)

    small_net = SmallNet()
    small_opt = torch.optim.Adam(small_net.parameters(), lr=1e-3)
    iterate_dataloader(trainloader, small_net, small_opt, train=True, softtarget=large_net)
    iterate_dataloader(testloader, small_net, small_opt, train=False)
