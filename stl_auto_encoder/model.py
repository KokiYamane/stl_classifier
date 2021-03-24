from torch import nn


class PcdAutoencoder(nn.Module):
    def __init__(self, input_num, hide_num, rep_num):
        super(PcdAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_num, hide_num),
            nn.ReLU(),
            nn.Linear(hide_num, hide_num),
            nn.ReLU(),
            nn.Linear(hide_num, rep_num),
        )

        self.decoder = nn.Sequential(
            nn.Linear(rep_num, hide_num),
            nn.ReLU(),
            nn.Linear(hide_num, hide_num),
            nn.ReLU(),
            nn.Linear(hide_num, input_num),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
