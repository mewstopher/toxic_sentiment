from torch.utils.data import Dataset
import pandas as pd
import logging
from enum import Enum
import pkg_resources
from pathlib import Path
from toxic_sentiment.exceptions import ScriptPathError, ScriptError
import subprocess


class Constants(Enum):
    GLOVE_SCRIPT = 'scripts/get_embeddings.sh'


class ToxicDataset(Dataset):

    def __init__(self, data_csv_path: str, glove_path: str):
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"{__name__} entered")
        self.data = pd.read_csv(data_csv_path)
        self.vocab = None

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class Embedding:

    def __init__(self):
        self.logger = logging.getLogger('Embedding')
        self.logger.debug("Embedding entered")

    def get_script(self):
        script = pkg_resources.resource_filename(
            'toxic_sentiment',
            Constants.GLOVE_SCRIPT.value
        )
        if Path(script).is_file():
            glove_script = script
        else:
            self.logger.error('No script found for getting embeddings found')
            raise ScriptPathError
        return glove_script

    def download_emb(self, download_path):
        glove_script = self.get_script()
        try:
            self.logger.info('Downloading Embeddings.. this may take a while')
            retval = subprocess.check_call(
                f'{glove_script} '
                f'{download_path}',
                shell=True
            )
        except subprocess.CalledProcessError as exc:
            raise ScriptError(str(exc))

