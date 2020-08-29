# _*_ coding: utf-8 _*_

"""Console script for toxic_sentiment."""
import sys
import click
from pathlib import Path
from toxic_sentiment.data_processors import ToxicDataset, Embedding
from toxic_sentiment.session import Session
from toxic_sentiment.models import BasicLstm
from logging.config import fileConfig

logging_config = Path.joinpath(Path(__file__).parent, 'logging.ini')
fileConfig(logging_config)


@click.group()
def main(args=None):
    """console script for toxic_sentiment."""
    return 0


@main.command()
@click.argument('path', type=str)
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
@click.argument('epochs', type=int, default=1)
@click.option('-s', '--save', is_flag=True)
def train_model(data_path, glove_path, save, epochs):
    toxic_dataset = ToxicDataset(data_path, glove_path)
    model = BasicLstm(toxic_dataset.embeddings, 200)
    session = Session(model, save=save)
    session.run(toxic_dataset, epochs)
    return 0


@main.command()
def predict():
    pass

if __name__ == "__main__":
    sys.exit(main()) # pragma: no cover
