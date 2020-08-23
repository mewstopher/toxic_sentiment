# _*_ coding: utf-8 _*_

"""Console script for toxic_sentiment."""
import sys
import click
from toxic_sentiment.data_processors import ToxicDataset, Embedding
from toxic_sentiment.session import Session
from logging.config import fileConfig
import logging

fileConfig('logging.ini')


@click.group()
def main(args=None):
    """console script for toxic_sentiment."""
    click.echo("Hello, what would you like to search for?")
    return 0


@main.command()
@click.option('-p', '--path', type=str)
def setup(path: str):
    try:
        glove_embedder = Embedding()
        glove_embedder.download_emb(path)
    except Exception as exc:
        click.secho(str(exc), fg='red', err=True)
    return 0


@main.command()
@click.argument('data_path', type=str)
@click.argument('glove_path', type=str)
def train_model(data_path, text_col, glove_path, vocab_path):
    train_logger = logging.getLogger('train_logger')
    toxic_dataset = ToxicDataset(data_path, glove_path)
    session = Session()
    session.train(toxic_dataset)
    train_sampler, val_sampler, test_sampler = train_test_sampler(toxic_dataset, .8, .1, .1)
    train_dataloader = DataLoader(toxic_dataset, batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(toxic_dataset, 128, sampler=val_sampler)
    test_dataloader = DataLoader(toxic_dataset, 128, sampler=test_sampler)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LstmNet(toxic_dataset.initial_embeddings, 200, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    BCELoss = nn.BCELoss()
    losses = {}
    num_epochs = 1
    count = 0
    session.train()
    return 0


@main.command()
def predict():
    pass

if __name__ == "__main__":
    sys.exit(main()) # pragma: no cover
