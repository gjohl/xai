from xai.models.simple_cnn import CNNBinaryClassifier2D
from xai.experiments.model_training.train_binary_classifier import run_model_training


model = CNNBinaryClassifier2D()
model_output_filename = "binary_cnn_mnist_2d_run_1.pth"
run_model_training(model, model_output_filename)
