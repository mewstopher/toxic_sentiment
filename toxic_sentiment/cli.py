# _*_ coding: utf-8 _*_

"""Console script for toxic_sentiment."""
import sys
import click
from toxic_sentiment.data_processors import ToxicDataset, Embedding
from logging.config import fileConfig

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
def get_data(data_path, glove_path):
    toxic_dataset = ToxicDataset(data_path, glove_path)
    return 0


if __name__ == "__main__":
    sys.exit(main()) # pragma: no cover
