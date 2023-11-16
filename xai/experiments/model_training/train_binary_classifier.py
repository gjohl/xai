from torch.utils.data import DataLoader

from xai.constants import MODEL_DIR
from xai.data_handlers.utils import load_training_data_mnist_binary
from xai.models.simple_cnn import BaseModel
from xai.models.training import Learner


def run_model_training(
        model,
        model_output_filename,
        batch_size=64,
        shuffle=True,
        train_validation_split=[0.8, 0.2],
        num_epochs=10
):
    train_dl, validation_dl = load_training_data_mnist_binary(batch_size, shuffle, train_validation_split)
    learn = train_model(model, train_dl, validation_dl, num_epochs)
    learn.save_model(MODEL_DIR / model_output_filename)


def train_model(model: BaseModel, train_dl: DataLoader, validation_dl: DataLoader, num_epochs: int):
    """Train a model on the given data"""
    learn = Learner(model, train_dl, validation_dl, num_epochs)
    learn.fit()
    return learn
