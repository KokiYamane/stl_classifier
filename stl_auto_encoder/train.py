import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from stl_auto_encoder.dataset import STLDataset
from stl_auto_encoder.model import PcdAutoencoder
import matplotlib.pyplot as plt


def train_net(n_epochs, train_loader, net, optimizer_cls=optim.Adam,
              loss_fn=nn.MSELoss(), device='cpu'):
    losses = []
    optimizer = optimizer_cls(net.parameters(), lr=0.001)
    net.to(device)

    for epoch in range(n_epochs):
        running_loss = 0.0
        net.train()

        for i, x in enumerate(train_loader):
            x.to(device)
            optimizer.zero_grad()
            XX_pred = net(x)
            loss = loss_fn(x, XX_pred)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        losses.append(running_loss / i)
        print('epoch', epoch, ': ', running_loss / i)

    return losses


def main():
    net = PcdAutoencoder(input_num=1500, hide_num=500, rep_num=100)
    datafolder = 'data'
    dataset = STLDataset(datafolder)
    print('length:', len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=10,
        pin_memory=True)
    losses = train_net(n_epochs=30,
                       train_loader=dataloader,
                       net=net)
    plt.plot(losses)
    plt.savefig('loss.png')


if __name__ == '__main__':
    main()
