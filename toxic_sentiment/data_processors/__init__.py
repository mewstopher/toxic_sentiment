from toxic_sentiment.data_processors.functions import tokenize_content, clean_special_char, tokenize_sample
from toxic_sentiment.exceptions import ScriptPathError, ScriptError, PathError
from torch.utils.data import Dataset
from pathlib import Path
from enum import Enum
import pkg_resources
import pandas as pd
import numpy as np
import subprocess
import logging
import torch
import json


class Constants(Enum):
    GLOVE_SCRIPT = 'scripts/get_embeddings.sh'
    UNK_WORD = 'unknown word'


class ToxicDataset(Dataset):

    def __init__(self, data_csv_path: str, text_col: str, glove_path: str,
                 vocab_path="../"):
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"{__name__} entered")
        self.df = pd.read_csv(data_csv_path)
        self.text_col = text_col
        self.unknown_words = []
        self.word_count = 0
        self.vocab_path = self.get_vocab_path(vocab_path)
        self.word_dicts = self.read_glove_vecs(glove_path)
        self.embeddings = self.get_initial_emb_dim()
        self.unk_encountered = False
        self.build_vocab()

    def get_initial_emb_dim(self):
        """
        get the initial embedding dimension
        """
        emb_dim = np.int(self.word_dicts['word_to_vec_map']['fox'].shape[0])
        emb_dim_vector = np.zeros(emb_dim, )
        return emb_dim_vector

    def get_vocab_path(self, path_to_vocab):
        """
        checks if path for vocab vectors to be stored
        is a valid directory

        """
        v_path = Path(path_to_vocab).is_dir()
        if not v_path:
            self.logger.error('{} is not a valid directory.'.format(v_path))
            raise PathError
        self.logger.debug('Using path: {} for vocab path'.format(v_path))
        return v_path

    def read_glove_vecs(self, glove_path: str) -> dict:
        """
        glove vec sare made up of a string of numbers
        space-separated with the corresponding word as
        the first 'number'
        returns a nested dictionary
        3 dictionaries needed:
            word_to_index, index_to_word, word_to_vec

        PARAMS
        -------
        glove_path: str
            path/to/glove/embeddings.txt
        """
        self.logger.debug('Creating word dictionaries')
        word_dicts = {}
        with open(glove_path, 'r') as f:
            words = set()
            word_dicts['word_to_vec_map'] = {}
            for line in f:
                line = line.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                word_dicts['word_to_vec_map'][curr_word] = \
                    np.array(line[1:], dtype=np.float64)
            i = 1
            word_dicts['word_to_index'] = {}
            word_dicts['index_to_word'] = {}
            for w in sorted(words):
                word_dicts['word_to_index'][w] = i
                word_dicts['index_to_word'][i] = w
                i += 1
        self.logger.debug('Word dicts created successfully')
        return word_dicts

    def build_vocab_vecs(self, tokenized_words: set) -> dict:
        embeddings = [self.embeddings]
        vocab = {}
        self.word_count = 0
        for word in tokenized_words:
            if word not in vocab:
                vocab[word] = self.word_count
                self.logger.debug('adding "{}" to vocab'.format(word))
                vectorized_word = self._vec(word)
                if vectorized_word != Constants.UNK_WORD.value:
                    embeddings.append(vectorized_word)
                    self.word_count += 1
            elif not self.unk_encountered:
                self.unk_index = self.word_count
                embeddings.append(self._vec_unk)
                self.unk_encountered = True
                self.word_count += 1
            else:
                vocab[word] = self.unk_index
            self.logger.debug('word count is: {}'.format(self.word_count))
        if not self.unk_encountered:
            embeddings.append(self._vec('unk'))
            self.unk_encountered = True
            self.word_count += 1
            vocab[Constants.UNK_WORD.value] = self.word_count
        vocab_vectors = {
            'embeddings': embeddings,
            'vocab': vocab
        }
        return vocab_vectors

    def build_vocab(self):
        """
        build a vocab dictionary that maps words
        to glove representations
        """
        tokenized_sample: set = tokenize_content(self.df['comment_text'])
        vocab_bundle = self.build_vocab_vecs(tokenized_sample)
        return vocab_bundle

    def _vec(self, word):
        try:
            word_as_vec = self.word_dicts['word_to_vec_map'][word]
        except KeyError:
            self.unknown_words.append(word)
            word_as_vec = Constants.UNK_WORD.value
        return word_as_vec

    @property
    def _vec_unk(self):
        """
        return vectorized unknown
        """
        return self._vec('unk')

    def __len__(self):
        return len(self.df.shape)

    def __getitem__(self, idx):
        sample_data = self.df.iloc[idx]
        text_tokenized = tokenize_sample(sample_data[self.text_col])
        text_indices = torch.tensor([self.vocab.get(i, self.unk_index) for i in text_tokenized], dtype=torch.long)
        text_len = len(text_indices)
        labels = torch.tensor(
            sample_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values.astype(float),
            dtype=torch.float32)
        text_padded = F.pad(text_indices, (0, self.max_text_len - text_len), value=0, mode='constant')
        return text_padded, labels


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
