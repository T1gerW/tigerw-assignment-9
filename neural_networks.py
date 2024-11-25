import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))
        
        # For visualization
        self.activations = {}
        self.gradients = {}

    def activate(self, Z):
        if self.activation_fn == 'tanh':
            return np.tanh(Z)
        elif self.activation_fn == 'relu':
            return np.maximum(0, Z)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        else:
            raise ValueError("Unsupported activation function")


    def forward(self, X):
        # Hidden layer computations
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activate(self.Z1)
        # Output layer computations
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = 1 / (1 + np.exp(-self.Z2))  # Sigmoid activation
        return self.A2



    def backward(self, X, y):
        m = X.shape[0]
        # Compute derivative of loss w.r.t. output
        dA2 = self.A2 - y  # Assuming y is {0,1} and using cross-entropy loss
        # Gradient w.r.t. W2 and b2
        dW2 = np.dot(self.A1.T, dA2) / m
        db2 = np.sum(dA2, axis=0, keepdims=True) / m
        # Backpropagate to hidden layer
        if self.activation_fn == 'tanh':
            dZ1 = np.dot(dA2, self.W2.T) * (1 - np.power(self.A1, 2))
        elif self.activation_fn == 'relu':
            dZ1 = np.dot(dA2, self.W2.T) * (self.A1 > 0)
        elif self.activation_fn == 'sigmoid':
            dZ1 = np.dot(dA2, self.W2.T) * (self.A1 * (1 - self.A1))
        # Gradient w.r.t. W1 and b1
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        # Update weights and biases
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        # Store gradients for visualization
        self.dW1 = dW1
        self.dW2 = dW2
    
    def activate(self, Z):
        if self.activation_fn == 'tanh':
            return np.tanh(Z)
        elif self.activation_fn == 'relu':
            return np.maximum(0, Z)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        else:
            raise ValueError("Unsupported activation function")



def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Current training step
    current_step = frame * 10  # Since we run 10 training steps per frame

    # Perform training steps
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)

    # Hidden Layer Feature Space Plot
    hidden_features = mlp.A1  # Shape: (n_samples, hidden_dim)
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                      c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title(f'Hidden Space at step {current_step}')
    ax_hidden.set_xlabel('Hidden Unit 1')
    ax_hidden.set_ylabel('Hidden Unit 2')
    ax_hidden.set_zlabel('Hidden Unit 3')

    # Plot the decision hyperplane in the hidden space
    # Create a grid of points in the hidden space
    h1_min, h1_max = hidden_features[:, 0].min() - 1, hidden_features[:, 0].max() + 1
    h2_min, h2_max = hidden_features[:, 1].min() - 1, hidden_features[:, 1].max() + 1
    h1_range = np.linspace(h1_min, h1_max, 20)
    h2_range = np.linspace(h2_min, h2_max, 20)
    h1_grid, h2_grid = np.meshgrid(h1_range, h2_range)

    # Compute h3 values for the hyperplane where Z2 = 0
    # Using the equation: W2[0]*h1 + W2[1]*h2 + W2[2]*h3 + b2 = 0
    # Solve for h3: h3 = (-W2[0]*h1 - W2[1]*h2 - b2) / W2[2]

    W2 = mlp.W2  # Shape: (hidden_dim, output_dim) = (3,1)
    b2 = mlp.b2  # Shape: (1, output_dim) = (1,1)
    # Flatten W2 to a 1D array
    W2_flat = W2.flatten()
    b2_scalar = b2.item()

    # Check if W2[2] is not zero to avoid division by zero
    if np.abs(W2_flat[2]) > 1e-6:
        # Suppress warnings for invalid values
        with np.errstate(divide='ignore', invalid='ignore'):
            h3_grid = (-W2_flat[0] * h1_grid - W2_flat[1] * h2_grid - b2_scalar) / W2_flat[2]
        # Mask out any NaN or infinite values
        h3_grid = np.ma.array(h3_grid, mask=np.isnan(h3_grid) | np.isinf(h3_grid))
        # Plot the hyperplane
        ax_hidden.plot_surface(h1_grid, h2_grid, h3_grid, alpha=0.3, color='yellow')
    else:
        # If W2[2] is zero, we can plot a vertical plane or skip plotting
        print(f"Skipping hyperplane plotting at step {current_step} due to W2[2] being zero")

    # Input Space Plot with Decision Boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = mlp.forward(grid)
    probs = probs.reshape(xx.shape)
    ax_input.contourf(xx, yy, probs, levels=[-1, 0, 1], cmap='bwr', alpha=0.2)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title(f'Input Space at step {current_step}')
    ax_input.set_xlabel('X1')
    ax_input.set_ylabel('X2')

    # Gradients Visualization
    ax_gradient.set_title(f'Gradients at step {current_step}')

    # Plot neurons as large blue dots
    # Positions for neurons
    nodes_input = [(0, 0), (1, 0)]  # x1 and x2
    nodes_hidden = [(-1, 1), (0, 1), (1, 1)]  # h1, h2, h3
    node_output = (0, 2)  # y

    # Plot input neurons
    for pos, label in zip(nodes_input, ['x1', 'x2']):
        ax_gradient.plot(pos[0], pos[1], 'bo', markersize=15)
        ax_gradient.text(pos[0], pos[1] - 0.2, label, ha='center', fontsize=12)

    # Plot hidden neurons
    for idx, pos in enumerate(nodes_hidden):
        ax_gradient.plot(pos[0], pos[1], 'bo', markersize=15)
        ax_gradient.text(pos[0], pos[1] + 0.1, f'h{idx + 1}', ha='center', fontsize=12)

    # Plot output neuron
    ax_gradient.plot(node_output[0], node_output[1], 'bo', markersize=15)
    ax_gradient.text(node_output[0], node_output[1] + 0.1, 'y', ha='center', fontsize=12)

    # Plot edges with thickness representing gradient magnitude
    # Edges from input to hidden layer
    for i, input_pos in enumerate(nodes_input):
        for j, hidden_pos in enumerate(nodes_hidden):
            weight = mlp.W1[i, j]
            gradient = mlp.dW1[i, j]
            linewidth = np.abs(gradient) * 100  # Scale for visibility
            color = 'g' if weight >= 0 else 'r'
            ax_gradient.plot([input_pos[0], hidden_pos[0]], [input_pos[1], hidden_pos[1]],
                             color=color, linewidth=linewidth)

    # Edges from hidden to output layer
    for j, hidden_pos in enumerate(nodes_hidden):
        weight = mlp.W2[j, 0]
        gradient = mlp.dW2[j, 0]
        linewidth = np.abs(gradient) * 100  # Scale for visibility
        color = 'g' if weight >= 0 else 'r'
        ax_gradient.plot([hidden_pos[0], node_output[0]], [hidden_pos[1], node_output[1]],
                         color=color, linewidth=linewidth)

    # Adjust plot
    ax_gradient.set_xlim(-2, 2)
    ax_gradient.set_ylim(-1, 3)
    ax_gradient.axis('off')



def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)
    plt.tight_layout()

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input,
                                     ax_hidden=ax_hidden, ax_gradient=ax_gradient,
                                     X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()


if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)