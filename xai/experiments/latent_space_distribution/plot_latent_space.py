from xai.constants import MODEL_DIR
from xai.models.simple_cnn import CNNBinaryClassifier2D

MODEL_FNAME = "binary_cnn_mnist_2d_run_1.pth"

model = CNNBinaryClassifier2D()
model.load(MODEL_DIR / MODEL_FNAME)

# Check accuracy
