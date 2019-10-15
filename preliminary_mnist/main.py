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
            nn.Linear(30, 10)
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.main(x)
        return x


def cat_crossentropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def iterate_dataloader(dataloader, net, optimizer, train, softtarget=None, temp=3):
    hard_criterion = torch.nn.CrossEntropyLoss()
    running_acc = 0
    batches = 0

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
            soft_loss = cat_crossentropy(out, soft_y)

        loss = hard_loss + soft_loss

        pred_labels = torch.argmax(out, dim=1)

        if train:
            loss.backward()
            optimizer.step()

        acc = torch.mean((pred_labels == y).type(torch.FloatTensor))
        running_acc += acc.item()
        batches = i
        pbar.set_postfix_str(f"{running_acc / batches * 100:.1f}% {hard_loss.item():.3f}HL {soft_loss.item():.3f}SL")

    pbar.close()
    return running_acc / batches


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    large_net = LargeNet()
    large_opt = torch.optim.SGD(large_net.parameters(), lr=1e-3)
    iterate_dataloader(trainloader, large_net, large_opt, train=True)
    acc = iterate_dataloader(testloader, large_net, large_opt, train=False)
    print(f"\nLarge Net with Hard Training: {100 * acc:.1f}%")

    small_net = SmallNet()
    small_opt = torch.optim.SGD(small_net.parameters(), lr=1e-3)
    iterate_dataloader(trainloader, small_net, small_opt, train=True)
    acc = iterate_dataloader(testloader, small_net, small_opt, train=False)
    print(f"\nSmall Net with Hard Training: {100 * acc:.1f}%")

    small_net = SmallNet()
    small_opt = torch.optim.SGD(small_net.parameters(), lr=1e-3)
    iterate_dataloader(trainloader, small_net, small_opt, train=True, softtarget=large_net, temp=3)
    acc = iterate_dataloader(testloader, small_net, small_opt, train=False)
    print(f"\nSmall Net with Soft Training: {100 * acc:.1f}%")
