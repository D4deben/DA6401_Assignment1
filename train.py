import argparse
import numpy as np
import matplotlib.pyplot as plt
import wandb
from keras.datasets import fashion_mnist

class FeedForward:
    def __init__(self, config):
        self.config = config
        self.layer_dims = self._create_architecture()
        self.theta = self._initialize_parameters()
        self.optimizer_state = {}
        (self.X_train, self.y_train), (self.X_val, self.y_val) = self._load_and_split_data()
        self._prepare_test_data()
        self._initialize_optimizer()

    def _create_architecture(self):
        layer_dims = [784]  # Input layer
        layer_dims += [self.config.hidden_size] * self.config.num_layers
        layer_dims.append(10)  # Output layer
        return layer_dims

    def _initialize_parameters(self):
        np.random.seed(42)
        theta = {}
        for l in range(1, len(self.layer_dims)):
            prev_dim = self.layer_dims[l-1]
            curr_dim = self.layer_dims[l]

            if self.config.weight_init == 'xavier':
                scale = np.sqrt(2.0/(prev_dim + curr_dim)) if self.config.activation == 'relu' \
                    else np.sqrt(1.0/prev_dim)
            else:
                scale = 0.01

            theta[f'W{l}'] = np.random.randn(curr_dim, prev_dim) * scale
            theta[f'b{l}'] = np.zeros((curr_dim, 1))
        return theta

    def _load_and_split_data(self):
        (X_train_full, y_train_full), (_, _) = fashion_mnist.load_data()
        m = X_train_full.shape[0]
        split = int(m * 0.9)

        X_train = X_train_full[:split].reshape(split, -1).T / 255.0
        y_train = np.eye(10)[y_train_full[:split]].T
        X_val = X_train_full[split:].reshape(m-split, -1).T / 255.0
        y_val = np.eye(10)[y_train_full[split:]].T

        return (X_train, y_train), (X_val, y_val)

    def _prepare_test_data(self):
        (_, _), (X_test, y_test) = fashion_mnist.load_data()
        self.X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0
        self.y_test = np.eye(10)[y_test].T

    def _activation(self, z, derivative=False):
        if self.config.activation == 'sigmoid':
            return self._sigmoid(z, derivative)
        elif self.config.activation == 'tanh':
            return self._tanh(z, derivative)
        elif self.config.activation == 'relu':
            return self._relu(z, derivative)

    @staticmethod
    def _sigmoid(z, derivative=False):
        if derivative:
            return z * (1 - z)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _tanh(z, derivative=False):
        if derivative:
            return 1 - z**2
        return np.tanh(z)

    @staticmethod
    def _relu(z, derivative=False):
        if derivative:
            return (z > 0).astype(float)
        return np.maximum(0, z)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward(self, X):
        fpass = {}
        A = X
        L = len(self.theta) // 2

        for l in range(1, L):
            fpass[f'a{l}'] = np.dot(self.theta[f'W{l}'], A) + self.theta[f'b{l}']
            fpass[f'h{l}'] = self._activation(fpass[f'a{l}'])
            A = fpass[f'h{l}']

        fpass[f'a{L}'] = np.dot(self.theta[f'W{L}'], A) + self.theta[f'b{L}']
        fpass['y_hat'] = self._softmax(fpass[f'a{L}'])
        return fpass

    def backward(self, X, Y, fpass):
        grads = {}
        L = len(self.theta) // 2
        m = X.shape[1]
        d_aL = fpass['y_hat'] - Y

        for l in range(L, 0, -1):
            A_prev = X if l == 1 else fpass[f'h{l-1}']
            grads[f'W{l}'] = np.dot(d_aL, A_prev.T)/m
            grads[f'b{l}'] = np.sum(d_aL, axis=1, keepdims=True)/m

            if self.config.weight_decay > 0:
                grads[f'W{l}'] += (self.config.weight_decay * self.theta[f'W{l}'])/m

            if l > 1:
                d_h = np.dot(self.theta[f'W{l}'].T, d_aL)
                d_aL = d_h * self._activation(fpass[f'h{l-1}'], derivative=True)

        return grads

    def compute_loss(self, y, y_hat):
        cross_entropy = -np.mean(np.sum(y * np.log(y_hat + 1e-9), axis=0))
        if self.config.weight_decay > 0:
            l2_penalty = sum(np.sum(np.square(w)) for w in self.theta.values() if w.ndim > 1)
            cross_entropy += (self.config.weight_decay * l2_penalty)/(2 * y.shape[1])
        return cross_entropy

    def _initialize_optimizer(self):
        opt = self.config.optimizer
        if opt in ['momentum', 'nesterov']:
            self.optimizer_state = {k: np.zeros_like(v) for k, v in self.theta.items()}
        elif opt in ['rmsprop']:
            self.optimizer_state = {k: np.zeros_like(v) for k, v in self.theta.items()}
        elif opt in ['adam', 'nadam']:
            self.optimizer_state = {
                'm': {k: np.zeros_like(v) for k, v in self.theta.items()},
                'v': {k: np.zeros_like(v) for k, v in self.theta.items()},
                't': 0
            }

    def _update_parameters(self, grads):
        opt = self.config.optimizer
        lr = self.config.lr
        beta = 0.9
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8

        if opt == 'sgd':
            for key in self.theta:
                self.theta[key] -= lr * grads[key]

        elif opt in ['momentum', 'nesterov']:
            for key in self.theta:
                self.optimizer_state[key] = beta * self.optimizer_state[key] + (1 - beta) * grads[key]
                self.theta[key] -= lr * self.optimizer_state[key]

                if opt == 'nesterov':
                    self.theta[key] -= lr * beta * self.optimizer_state[key]

        elif opt == 'rmsprop':
            for key in self.theta:
                self.optimizer_state[key] = beta * self.optimizer_state[key] + (1 - beta) * np.square(grads[key])
                self.theta[key] -= lr * grads[key] / (np.sqrt(self.optimizer_state[key]) + epsilon)

        elif opt in ['adam', 'nadam']:
            self.optimizer_state['t'] += 1
            m = self.optimizer_state['m']
            v = self.optimizer_state['v']

            for key in self.theta:
                m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
                v[key] = beta2 * v[key] + (1 - beta2) * np.square(grads[key])

                m_hat = m[key] / (1 - beta1**self.optimizer_state['t'])
                v_hat = v[key] / (1 - beta2**self.optimizer_state['t'])

                if opt == 'nadam':
                    m_hat = beta1 * m_hat + (1 - beta1) * grads[key]

                self.theta[key] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

    def train(self):
        X, Y = self.X_train, self.y_train
        m = X.shape[1]
        batch_size = self.config.batch_size or m
        steps_per_epoch = m // batch_size

        for epoch in range(self.config.epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]

            epoch_loss = 0
            for step in range(steps_per_epoch):
                start = step * batch_size
                end = start + batch_size
                X_batch = X_shuffled[:, start:end]
                Y_batch = Y_shuffled[:, start:end]

                fpass = self.forward(X_batch)
                grads = self.backward(X_batch, Y_batch, fpass)
                self._update_parameters(grads)

                epoch_loss += self.compute_loss(Y_batch, fpass['y_hat'])

            # Validation
            val_fpass = self.forward(self.X_val)
            val_loss = self.compute_loss(self.y_val, val_fpass['y_hat'])
            val_acc = self.accuracy(self.X_val, self.y_val)

            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_loss/steps_per_epoch,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

    def accuracy(self, X, y):
        fpass = self.forward(X)
        predictions = np.argmax(fpass['y_hat'], axis=0)
        labels = np.argmax(y, axis=0)
        return np.mean(predictions == labels)
def main(config):
    wandb.init( )
    config = wandb.config

    # Create meaningful run name
    wandb.run.name = (
        f"D4deben_hl{config.num_layers}_bs{config.batch_size}_"
        f"{config.activation[:3]}_lr{config.lr}_"
        f"{config.optimizer[:3]}_wd{config.weight_decay}"
    )

    model = FeedForward(config)
    model.train()

    test_acc = model.accuracy(model.X_test, model.y_test)
    wandb.log({"test_acc": test_acc})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", default="myprojectname", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", default="myname", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d", "--dataset", default="fashion_mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-l", "--loss", default="cross_entropy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("-o", "--optimizer", default="adam", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-m", "--momentum", type=float, default=0.9)
    parser.add_argument("-beta", "--beta", type=float, default=0.999)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999)
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0005)
    parser.add_argument("-w_i", "--weight_init", default="Xavier", choices=["random", "Xavier"])
    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument("-sz", "--hidden_size", type=int, default=128)
    parser.add_argument("-a", "--activation", default="relu", choices=["identity", "sigmoid", "tanh", "ReLU"])
    
    config = parser.parse_args()
    main(config)
