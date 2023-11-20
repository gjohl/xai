from xai.models.simple_cnn import CNNBinaryClassifier3D
from xai.models.training.train_binary_classifier import run_model_training


model = CNNBinaryClassifier3D()
model_output_filename = "binary_cnn_mnist_3d_run_1.pth"
run_model_training(model, model_output_filename)
