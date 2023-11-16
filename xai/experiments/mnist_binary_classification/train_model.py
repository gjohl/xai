from xai.models.simple_cnn import CNNBinaryClassifier
from xai.experiments.model_training.train_binary_classifier import run_model_training


model = CNNBinaryClassifier()
model_output_filename = "binary_cnn_mnist_run_1.pth"
run_model_training(model, model_output_filename)
