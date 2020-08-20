# _*_ coding: utf-8 _*_

"""Console script for toxic_sentiment."""
import sys
import click
from toxic_sentiment.toxic_sentiment import func1


@click.command()
def main(args=None):
    """console script for toxic_sentiment."""
    click.echo("Hello, what would you like to search for?")
    return 0


if __name__ == "__main__":
    sys.exit(main()) # pragma: no cover
