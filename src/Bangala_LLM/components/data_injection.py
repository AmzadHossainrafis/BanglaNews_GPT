import torch
from bnlp import CleanText

clear = CleanText(
    fix_unicode=True,
    unicode_norm=True,
    unicode_norm_form="NFKC",
    remove_url=True,
    remove_email=False,
    remove_emoji=False,
    remove_number=True,
    remove_digits=False,
    remove_punct=False,
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_number="<NUMBER>",
    replace_with_digit="<DIGIT>",
    replace_with_punct="<PUNC>",
)


class DataInjector:
    """
    This class is responsible for spliting the data into train and val set
    """

    def __init__(self, data_dir, train_size=0.8, clean_text=clear):
        self.data_dir = data_dir
        self.train_size = train_size
        self.clean_text = clean_text

    def split_data(self):
        with open(self.data_dir, "r") as f:
            data = f.read()

        data = [self.clean_text(d) for d in data]
        train_size = int(len(data) * self.train_size)
        train_data = data[:train_size]
        val_data = data[train_size:]
        return train_data, val_data

    def save_data(self, train_data, val_data, train_path, val_path):
        with open(train_path, "w") as f:
            f.write("\n".join(train_data))

        with open(val_path, "w") as f:
            f.write("\n".join(val_data))

    def clean_text(self, data):
        clean_data = self.clean_text(data)
        return clean_data
