import numpy as np
import utils
import typing
np.random.seed(1)


# Sigmoid functions
def sigmoid(x: np.ndarray): return 1/(1+np.exp(-x))
def sigmoid_differentiated(x: np.ndarray): return sigmoid(x)*(1-sigmoid(x))
def improved_sigmoid(x: np.ndarray): return 1.7159 * np.tanh((2/3)*x)
def improved_sigmoid_differentiated(x: np.ndarray): return 1.7159 * (2/3) * (1 - np.tanh((2/3)*x) ** 2)


# Softmax
def softmax(x): return np.exp(x)/np.transpose(np.array([np.sum(np.exp(x), axis=1)]))


def pre_process_images(X: np.ndarray, mean=33.55274553571429, std=78.87550070784701):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    if not mean:
        mean = np.mean(X)
    if not std:
        std = np.std(X)
    # print(f"Task 2a report: mean = {mean}, standard deviation = {std}")
    # Setting mean and standard deviation as we were to have the same values for each set

    # Implementing the equation 4 from the assignment
    X_norm = (X-mean)/std

    # Bias trick
    bias = np.ones((X.shape[0], 1))
    # X = np.append(X_norm, bias, axis=1)
    X = np.hstack((X_norm, bias))  # hstack like append with axis=1 default
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: y, labels/targets of each image of shape: [batch size, num_classes]
        outputs: y_hat, outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    # TODO: Implement this function (copy from last assignment)
    loss = -np.mean(np.sum(targets*np.log(outputs), axis=-1))
    assert targets.shape == outputs.shape, f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    return loss


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785  # (28*28+1)
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # TODO: If use_improved_weight_init is implemented here
        # Initialize the w
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if use_improved_weight_init:
                fan_in = 1/np.sqrt(prev)
                w = np.random.normal(0, fan_in, w_shape)  # mean = 0
            else:
                w = np.random.uniform(-1, 1, w_shape)
            self.ws.append(w)
            prev = size

        self.grads = [None for _ in range(len(self.ws))]
        self.hidden_layer_output = [None for _ in range(len(neurons_per_layer)-1)]
        self.zs = [None for _ in range(len(neurons_per_layer)-1)]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...

        # First layer
        z = X @ self.ws[0]
        self.zs[0] = z
        a = improved_sigmoid(z) if self.use_improved_sigmoid else sigmoid(z)
        self.hidden_layer_output[0] = a

        # Hidden layer(s)
        for layer in range(1, len(self.hidden_layer_output)):
            # Iterate through all hidden layers
            z = self.hidden_layer_output[layer-1] @ self.ws[layer]  # first hidden layer (1) will take first activated value
            self.zs[layer] = z
            a = improved_sigmoid(z) if self.use_improved_sigmoid else sigmoid(z)
            self.hidden_layer_output[layer] = a

        # Softmax of last hidden-layer activation output matrix-multiplied with the weigths
        z = self.hidden_layer_output[-1] @ self.ws[-1]
        output = softmax(z)
        return output

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        N = X.shape[0]  # batch size

        # Last layer
        delta = -(targets - outputs)  # Hint 2
        output_gradient = delta.T @ self.hidden_layer_output[-1] / N
        self.grads[-1] = output_gradient.T
        sig_dif = improved_sigmoid_differentiated if self.use_improved_sigmoid else sigmoid_differentiated

        # For hidden layers in reverse order (not include [0])
        for i in range(len(self.hidden_layer_output)-1, 0, -1):
            # delta update is from "show that" in task 1a
            delta = sig_dif(self.zs[i])*(delta@self.ws[i+1].T)
            self.grads[i] = self.hidden_layer_output[i-1].T@delta/N

        # First layer
        delta = sig_dif(self.zs[0])*(delta@self.ws[1].T)
        self.grads[0] = X.T@delta/N

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape, f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for _ in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    num_examples = Y.shape[0]
    encoded_array = np.zeros((num_examples, num_classes))
    for i, target in enumerate(Y):
        encoded_array[i][target] = 1
    return encoded_array


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited.
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
