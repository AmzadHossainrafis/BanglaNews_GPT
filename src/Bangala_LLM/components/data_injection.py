from bnlp import CleanText
from Bangala_LLM.utils.logger import logger

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
        logger.info("Splitting data into train and validation sets.")
        with open(self.data_dir, "r") as f:
            data = f.read()

        logger.info("Cleaning text data.")

        data = [self.clean_text(d) for d in data]
        train_size = int(len(data) * self.train_size)
        train_data = data[:train_size]
        val_data = data[train_size:]
        logger.info(
            f"Data split into {len(train_data)} training samples and {len(val_data)} validation samples."
        )
        return train_data, val_data

    def save_data(self, train_data, val_data, train_path, val_path):
        with open(train_path, "w") as f:
            f.write("\n".join(train_data))
            logger.info(f"Training data saved to {train_path}")

        with open(val_path, "w") as f:
            f.write("\n".join(val_data))
            logger.info(f"Validation data saved to {val_path}")

    def clean_text(self, data):

        clean_data = self.clean_text(data)
        return clean_data


if __name__ == "__main__":
    data_injector = DataInjector(
        data_dir="/home/amzad/Desktop/BanglaNews_GPT/dataset/shakespeare.txt"
    )
    train_data, val_data = data_injector.split_data()
    data_injector.save_data(train_data, val_data, "data/train.txt", "data/val.txt")
