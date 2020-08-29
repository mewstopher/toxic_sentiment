from toxic_sentiment.data_processors.functions import tokenize_content, clean_special_char, tokenize_sample
from toxic_sentiment.exceptions import ScriptPathError, ScriptError, PathError
from torch.utils.data import Dataset
from decouple import config
from pathlib import Path
from enum import Enum
import pkg_resources
from torch import nn
import pandas as pd
import numpy as np
import subprocess
import logging
import torch
import json


class Constants(Enum):
    GLOVE_SCRIPT = 'scripts/get_embeddings.sh'
    UNK_WORD = 'unknown word'
    TEXT_COL = 'comment_text'


class ToxicDataset(Dataset):

    def __init__(self, data_csv_path: str, glove_path: str):
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"{__name__} entered")
        self.df = pd.read_csv(data_csv_path)
        self.text_col = Constants.TEXT_COL.value
        self.unknown_words = []
        self.unk_index = None
        self.word_count = 0
        self.word_dicts = self.read_glove_vecs(glove_path)
        self.vocab = None
        self.embeddings = self.get_initial_emb_dim()
        self.unk_encountered = False
        self.build_vocab()
        self.max_text_len = len(max(self.df.comment_text))

    def get_initial_emb_dim(self):
        """
        get the initial embedding dimension
        """
        emb_dim = np.int(self.word_dicts['word_to_vec_map']['fox'].shape[0])
        emb_dim_vector = np.zeros(emb_dim, )
        return emb_dim_vector

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
        self.logger.info('Creating word dictionaries')
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
        self.logger.info('Word dicts created successfully')
        return word_dicts

    def save_vocab(self, vocab_bundle):
        vocab_path = config('vocab_path')
        embeddings_path = config('embeddings_path')
        if Path(vocab_path).is_dir() and Path(embeddings_path).is_dir():
            vocab_file = Path.joinpath(Path(embeddings_path), Constants.VOCAB_FILE.value)
            embeddings_file = Path.joinpath(Path(embeddings_path), Constants.EMB_FILE.value)
            np.save(vocab_file, vocab_bundle['vocab'])
            np.save(embeddings_file, vocab_bundle['embeddings'])
        else:
            self.logger.info('Cant save vocab. check vocab and embedding path in .ENV file')

    def build_vocab_vecs(self, tokenized_words: list) -> dict:
        embeddings = [self.embeddings]
        vocab = {}
        self.word_count = 0
        self.logger.info('Building vocabulary')
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
        vocab_loaded = self.load_vocab()
        if not vocab_loaded:
            tokenized_sample: list = tokenize_content(self.df['comment_text'])
            vocab_bundle = self.build_vocab_vecs(tokenized_sample)
            self.embeddings = vocab_bundle['embeddings']
            self.vocab = vocab_bundle['vocab']
            self.save_vocab(vocab_bundle)

    def load_vocab(self) -> bool:
        vocab_path = config('vocab_path')
        embeddings_path = config('embeddings_path')
        valid_v_path = Path(vocab_path).is_file()
        valid_emb_path = Path(embeddings_path).is_file()
        if valid_v_path and valid_emb_path:
            self.logger.info('using pre-built vocab and embeddings')
            self.vocab = np.load(vocab_path, allow_pickle=True).item()
            self.embeddings = np.load(embeddings_path, allow_pickle=True)
            self.unk_index = self.vocab['unk']
            found_vocab = True
        else:
            self.logger.info('Vocab or embeddings path not specified')
            found_vocab = False
        return found_vocab

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
        return len(self.df)

    def __getitem__(self, idx):
        sample_data = self.df.iloc[idx]
        text_tokenized = tokenize_sample(sample_data[self.text_col])
        text_indices = torch.tensor([self.vocab.get(i, self.unk_index) for i in text_tokenized], dtype=torch.long)
        text_len = len(text_indices)
        labels = torch.tensor(
            sample_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values.astype(float),
            dtype=torch.float32)
        text_padded = nn.functional.pad(text_indices, (0, self.max_text_len - text_len), value=0, mode='constant')
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
