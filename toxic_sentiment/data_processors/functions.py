import pandas as pd
from torch.utils.data import SubsetRandomSampler


def tokenize_content(text_col: pd.Series) -> list:
    """
    Tokenize all the rows of a column of
    text data from a dataframe

    PARAMS
    ---------
    text_col: df.Column
        text column of a dataset
    """
    text_cleaned = text_col.astype(str).apply(lambda x:
                                              clean_special_char(x).lower().split())
    text_tokenized = [word for sentence in text_cleaned for word in sentence]
    return text_tokenized


def tokenize_sample(sample_content):
    """
    clean and tokenize a single comment
    """
    cleaned_sample = clean_special_char(sample_content).lower().split()
    return cleaned_sample


def clean_special_char(text: str) -> str:
    punc = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    for p in punc:
        text = text.replace(p, '')
    return text


def train_test_sampler(dataset, train_split, val_split, test_split):
    """
    Returns 3 initialized classes for train, val, and test splits

    PARAMS
    -----------------
    dataset: Pytorch dataset class
    train_split: percent of data that should be in train
    val_split: percent of data to be in validation fold
    test_split: percent of data to be in test fold
    """
    dataset_len = dataset.__len__()
    dataset_indices = list(range(dataset_len))
    train_stop = int(train_split*dataset_len)
    val_stop = int(val_split*dataset_len) + train_stop
    test_stop = int(val_split*dataset_len)
    train_indices = dataset_indices[:train_stop]
    val_indices = dataset_indices[train_stop: val_stop]
    test_indices = dataset_indices[val_stop:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    return train_sampler, val_sampler, test_sampler
