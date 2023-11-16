from matplotlib import pyplot as plt
import torch
import torchvision

from xai.constants import MODEL_DIR
from xai.models.simple_cnn import CNNBinaryClassifier2D
from xai.data_handlers.utils import load_training_data_mnist_binary, load_test_data_mnist_binary
from xai.data_handlers.mnist import DEFAULT_MNIST_NORMALIZATION
from xai.evaluation_metrics.distance import SimplexDistance

BATCH_SIZE = 64
MODEL_FNAME = "binary_cnn_mnist_2d_run_1.pth"

model = CNNBinaryClassifier2D()
model.load(MODEL_DIR / MODEL_FNAME)


# ---------------Load data---------------
train_dl, validation_dl = load_training_data_mnist_binary(batch_size=BATCH_SIZE,
                                                          digits=(0, 1, 2),
                                                          shuffle=False,
                                                          train_validation_split=[0.8, 0.2])
test_dl = load_test_data_mnist_binary(batch_size=BATCH_SIZE,
                                      digits=(0, 1, 2),
                                      count_per_digit={0: 50, 1: 50, 2: 50},
                                      shuffle=False)
source_data, source_labels = next(iter(train_dl))
validation_data, validation_labels = next(iter(validation_dl))
test_data, test_labels = next(iter(test_dl))


# ---------------Plot different digits in latent space---------------
digits = (0, 1, 7)
n = 20

def get_digit_mask(labels, digit, n):
    """Return a boolean mask for the first n of the given digit."""
    label_mask = labels == digit
    count_mask = torch.cumsum(label_mask, 0) <= n
    idx_mask = label_mask & count_mask
    return idx_mask

# Load data by original label (digit) rather than boolean
input_data = test_dl.dataset.dataset.data
input_data = torchvision.transforms.Normalize(*DEFAULT_MNIST_NORMALIZATION)(input_data / 255)
input_data = input_data[:, None, :, :]  # A 1 dimension went missing somewhere?
labels = test_dl.dataset.dataset.targets

# Select a subset of points to plot
input_data_list = []
label_list = []
for digit in digits:
    digit_mask = get_digit_mask(labels, digit, n)
    input_data_list.append(input_data[digit_mask])
    label_list.append(labels[digit_mask])

input_data_all = torch.cat(input_data_list)
labels_all = torch.cat(label_list)

# Calculate latents
latents = model.latent_representation(input_data_all).detach()

# Create a scatter plot for each unique label
plt.figure()

for digit in digits:
    # Plot each label one-by-one
    x_data = latents[labels_all == digit, 0]
    y_data = latents[labels_all == digit, 1]
    plt.scatter(x_data, y_data, alpha=0.5, label=f'Digit {digit}')

plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title("Input digits in the model's latent space")
plt.legend()
plt.show()


# ---------------Plot residual shift---------------
# Fit simplex model
sd = SimplexDistance(model, source_data, input_data_all)
sd.distance()

# Get residual coordinate pairs
target_latents_approx = sd.simplex.latent_approx()
latents[0]
target_latents_approx[0]

# Plot line segments / movement
color_map = {
    0: 'b',
    1: 'r',
    7: 'g'
}
keep_n = 2  # None to keep all
plt.figure()
for digit in digits:
    # Plot each label one-by-one
    x_mask = [labels_all == digit, 0]
    y_mask = [labels_all == digit, 1]
    # true_start_x =

    x_values = [latents[x_mask][:keep_n], target_latents_approx[x_mask][:keep_n]]
    y_values = [latents[y_mask][:keep_n], target_latents_approx[y_mask][:keep_n]]
    plt.plot(x_values[0], y_values[0], 'bo', c=color_map[digit],  linestyle="--", alpha=0.5)
    plt.plot(x_values[1], y_values[1], 'bx', c=color_map[digit],  linestyle="--", alpha=0.5)
    # plt.legend()

plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title("Input digit's movement under simplex")
plt.show()




plt.figure()
# Coordinates of the start and end points
start_point = (1, 2)
end_point = (3, 4)

# Plot the line with a circular marker at the start and a triangle marker at the end
plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], linestyle='-', color='blue')
plt.plot(start_point[0], start_point[1], marker='o', color='blue')  # Triangle marker at the end
plt.plot(end_point[0], end_point[1], marker='^', color='blue')  # Triangle marker at the end

# Annotate the points
plt.text(start_point[0] - 0.1, start_point[1] + 0.2, "Start")
plt.text(end_point[0] - 0.1, end_point[1] - 0.2, "End")

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot with Circular and Triangle Markers')

# Show the plot
plt.show()