import pandas as pd


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
