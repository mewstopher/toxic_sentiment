from torch import nn
import torch


class BasicLstm(nn.Module):

    def __init__(self, embeddings, num_lstm_units, device, frozen=True, freeze_embeddings=True):
        super(BasicLstm, self).__init__()
        self.device = device
        self.embeddings = nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings),
            freeze=freeze_embeddings)
        self.num_lstm_units = num_lstm_units
        self.embeddings_dim = embeddings.shape[1]
        self.bilstm = nn.LSTM(input_size=self.embeddings_dim,
                              hidden_size=self.num_lstm_units,
                              batch_first=True, bidirectional=True)
        self.dense1 = nn.Linear(800, 128)
        self.dense1_dropout = nn.Dropout(0.5)
        self.dense_2 = nn.Linear(128, 64)
        self.dense2_dropout = nn.Dropout(0.5)
        self.output = nn.Linear(64,6)
        self.to(device)

    def init_hidden(self, batch_size):
        """
        Initializes Hidden state, cell state
        number-of-layers*number-ofdirections, batch_size, number
        of lstm units
        """
        hidden_a = torch.randn(1*2, batch_size, self.num_lstm_units).to(
            self.device)
        hidden_b = torch.randn(1*2, batch_size, self.num_lstm_units).to(
            self.device)

    def forward(self, x):
        """
        define forward propogation here
        """
        batch_size = x.shape[0]
        self.hidden = self.init_hidden(batch_size)
        x = self.embeddings(x).float()
        x, self.hidden = self.bilstm(x, self.hidden)
        x1 = torch.mean(x, 1)
        x2, _ = torch.max(x, 1)
        x = torch.cat((x1, x2), 1)
        x = nn.functional.relu(self.dense_1(x))
        x = self.dense1_dropout(x)
        x = nn.functional.relu(self.dense_2(x))
        x = self.dense2_dropout(x)
        x = nn.functional.sigmoid(self.output(x))
        return x

