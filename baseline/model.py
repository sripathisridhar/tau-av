from torch import nn


class BaselineModel(nn.Module):

    def __init__(self, emb_dim, num_classes):
        super(BaselineModel, self).__init__()

        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Linear(self.emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):

        y = self.model(x)
        return y