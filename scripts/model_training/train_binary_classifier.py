import click
from torch.utils.data import DataLoader

from xai.constants import MODEL_DIR
from xai.data_handlers.utils import load_training_data_mnist_binary
from xai.models.simple_cnn import CNNBinaryClassifier, BaseModel
from xai.models.training import Learner


@click.command()
@click.option("--model_filename", help="Filename to save the trained model.")
@click.option("--batch_size", help="Filename to save the trained model.")
@click.option("--num_epochs", help="Filename to save the trained model.")
def run_model_training(
        model_filename,
        batch_size=64,
        shuffle=True,
        train_validation_split=[0.8, 0.2],
        num_epochs=20
):
    model = CNNBinaryClassifier()
    train_dl, validation_dl = load_training_data_mnist_binary(batch_size, shuffle, train_validation_split)
    learn = train_model(model, train_dl, validation_dl, num_epochs)
    learn.save_model(MODEL_DIR / model_filename)


def train_model(model: BaseModel, train_dl: DataLoader, validation_dl: DataLoader, num_epochs: int):
    """Train a model on the given data"""
    learn = Learner(model, train_dl, validation_dl, num_epochs)
    learn.fit()
    return learn


if __name__ == "__main__":
    run_model_training()
