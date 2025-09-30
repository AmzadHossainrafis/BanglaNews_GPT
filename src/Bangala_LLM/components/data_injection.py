# from bnlp import CleanText
from Bangala_LLM.utils.logger import logger
from Bangala_LLM.utils.common import read_config
config = read_config("../../../config/config.yaml")


class DataLoaderInjectorv2: 
    def __init__(self, data_path ,config=None):
        self.data_path = data_path
        self.config = config

    def load_data(self):
        logger.info(f"Loading data from {self.data_path}")
        with open(self.data_path, "r") as f:
            data = f.read()
        return data
    
    def get_data(self):
        data = self.load_data()
        return data
    def split_data(self, data, split_ratio=0.8):
        logger.info("Splitting data into train and validation sets.")
        data = data.split("\n")
        train_size = int(len(data) * split_ratio)
        train_data = data[:train_size]
        val_data = data[train_size:]
        logger.info(
            f"Data split into {len(train_data)} training samples and {len(val_data)} validation samples."
        )
        #save train and val data to text file
        with open("/home/amzad/Desktop/BanglaNews_GPT/dataset/train.txt", "w") as f:
            f.write("\n".join(train_data))
            logger.info(f"Training data saved to dataset/train.txt")
        with open("/home/amzad/Desktop/BanglaNews_GPT/dataset/val.txt", "w") as f:
            f.write("\n".join(val_data))
            logger.info(f"Validation data saved to dataset/val.txt")
        return train_data, val_data
        


if __name__ == "__main__":
    data_injector = DataLoaderInjectorv2(
        data_path="/home/amzad/Desktop/BanglaNews_GPT/dataset/shakespeare.txt"
    )
    data = data_injector.get_data()
    train_data, val_data = data_injector.split_data(data)
    #data_injector.save_data(train_data, val_data, "dataset/train.txt", "dataset/val.txt")

# if __name__ == "__main__":
#     data_injector = DataInjector(
#         data_dir="/home/amzad/Desktop/BanglaNews_GPT/dataset/shakespeare.txt"
#     )
#     train_data, val_data = data_injector.split_data()
#     data_injector.save_data(train_data, val_data, "data/train.txt", "data/val.txt")
