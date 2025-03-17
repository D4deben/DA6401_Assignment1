import argparse
import wandb
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist

# Activation functions and their derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(h):
    return h * (1 - h)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(h):
    return 1 - h**2

def relu(z):
    return np.maximum(0, z)

def relu_derivative(h):
    return (h > 0).astype(float)

def identity(z):
    return z

def identity_derivative(h):
    return np.ones_like(h)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Loss functions
def cross_entropy_loss(y, y_hat):
    return -np.mean(np.sum(y * np.log(y_hat + 1e-9), axis=0))

def mean_squared_error(y, y_hat):
    return np.mean(np.sum((y - y_hat)**2, axis=0))

class NeuralNetwork:
    def __init__(self, config):
        self.config = config
        self.layers = self._create_layers()
        self.parameters = self._initialize_parameters()
        self.optimizer_state = {}
    
    def _create_layers(self):
        return [self.config.hidden_size] * self.config.num_layers + [10]
    
    def _initialize_parameters(self):
        parameters = {}
        input_size = 784
        for i, size in enumerate(self.layers):
            if self.config.weight_init == "Xavier":
                scale = np.sqrt(2.0 / (input_size + size))
            else:
                scale = 0.01
            parameters[f'W{i+1}'] = np.random.randn(size, input_size) * scale
            parameters[f'b{i+1}'] = np.zeros((size, 1))
            input_size = size
        return parameters
    
    def _get_activation(self, name):
        activations = {
            "sigmoid": (sigmoid, sigmoid_derivative),
            "tanh": (tanh, tanh_derivative),
            "ReLU": (relu, relu_derivative),
            "identity": (identity, identity_derivative)
        }
        return activations[name]
    
    def forward(self, X):
        activation, _ = self._get_activation(self.config.activation)
        A = X
        caches = []
        for i in range(len(self.layers) - 1):
            Z = np.dot(self.parameters[f'W{i+1}'], A) + self.parameters[f'b{i+1}']
            A = activation(Z)
            caches.append((A, Z))
        Z_out = np.dot(self.parameters[f'W{len(self.layers)}'], A) + self.parameters[f'b{len(self.layers)}']
        A_out = softmax(Z_out)
        caches.append((A_out, Z_out))
        return A_out, caches
    
    def backward(self, X, Y, caches):
        grads = {}
        m = X.shape[1]
        A_out, Z_out = caches[-1]
        dZ = A_out - Y
        
        for i in reversed(range(len(self.layers))):
            grads[f'W{i+1}'] = np.dot(dZ, caches[i-1][0].T if i > 0 else X.T) / m
            grads[f'b{i+1}'] = np.sum(dZ, axis=1, keepdims=True) / m
            if i > 0:
                _, activation_derivative = self._get_activation(self.config.activation)
                dA = np.dot(self.parameters[f'W{i+1}'].T, dZ)
                dZ = dA * activation_derivative(caches[i-1][1])
        
        return grads
    
    def update_parameters(self, grads):
        if self.config.optimizer == "sgd":
            for key in self.parameters:
                self.parameters[key] -= self.config.learning_rate * grads[key]
        
        elif self.config.optimizer == "momentum":
            if 'v' not in self.optimizer_state:
                self.optimizer_state['v'] = {k: np.zeros_like(v) for k, v in self.parameters.items()}
            for key in self.parameters:
                self.optimizer_state['v'][key] = (self.config.momentum * self.optimizer_state['v'][key] + 
                                                  self.config.learning_rate * grads[key])
                self.parameters[key] -= self.optimizer_state['v'][key]
        
        elif self.config.optimizer == "nag":
            if 'v' not in self.optimizer_state:
                self.optimizer_state['v'] = {k: np.zeros_like(v) for k, v in self.parameters.items()}
            for key in self.parameters:
                v_prev = self.optimizer_state['v'][key]
                self.optimizer_state['v'][key] = self.config.momentum * v_prev + self.config.learning_rate * grads[key]
                self.parameters[key] -= self.config.momentum * v_prev + (1 - self.config.momentum) * self.optimizer_state['v'][key]
        
        elif self.config.optimizer == "rmsprop":
            if 's' not in self.optimizer_state:
                self.optimizer_state['s'] = {k: np.zeros_like(v) for k, v in self.parameters.items()}
            for key in self.parameters:
                self.optimizer_state['s'][key] = (self.config.beta * self.optimizer_state['s'][key] + 
                                                  (1 - self.config.beta) * np.square(grads[key]))
                self.parameters[key] -= (self.config.learning_rate / (np.sqrt(self.optimizer_state['s'][key]) + self.config.epsilon)) * grads[key]
        
        elif self.config.optimizer in ["adam", "nadam"]:
            if 'm' not in self.optimizer_state:
                self.optimizer_state['m'] = {k: np.zeros_like(v) for k, v in self.parameters.items()}
                self.optimizer_state['v'] = {k: np.zeros_like(v) for k, v in self.parameters.items()}
                self.optimizer_state['t'] = 0
            
            self.optimizer_state['t'] += 1
            for key in self.parameters:
                self.optimizer_state['m'][key] = (self.config.beta1 * self.optimizer_state['m'][key] + 
                                                  (1 - self.config.beta1) * grads[key])
                self.optimizer_state['v'][key] = (self.config.beta2 * self.optimizer_state['v'][key] + 
                                                  (1 - self.config.beta2) * np.square(grads[key]))
                m_hat = self.optimizer_state['m'][key] / (1 - self.config.beta1 ** self.optimizer_state['t'])
                v_hat = self.optimizer_state['v'][key] / (1 - self.config.beta2 ** self.optimizer_state['t'])
                
                if self.config.optimizer == "adam":
                    self.parameters[key] -= self.config.learning_rate * m_hat / (np.sqrt(v_hat) + self.config.epsilon)
                else:  # nadam
                    m_hat_next = self.config.beta1 * m_hat + (1 - self.config.beta1) * grads[key] / (1 - self.config.beta1 ** self.optimizer_state['t'])
                    self.parameters[key] -= self.config.learning_rate * m_hat_next / (np.sqrt(v_hat) + self.config.epsilon)
    
    def train(self, X_train, y_train, X_val, y_val):
        for epoch in range(self.config.epochs):
            for i in range(0, X_train.shape[1], self.config.batch_size):
                X_batch = X_train[:, i:i+self.config.batch_size]
                y_batch = y_train[:, i:i+self.config.batch_size]
                
                A_out, caches = self.forward(X_batch)
                grads = self.backward(X_batch, y_batch, caches)
                self.update_parameters(grads)
            
            train_loss = self._compute_loss(X_train, y_train)
            val_loss = self._compute_loss(X_val, y_val)
            val_accuracy = self._compute_accuracy(X_val, y_val)
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })
            
            print(f"Epoch {epoch+1}/{self.config.epochs}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}")
    
    def _compute_loss(self, X, y):
        A_out, _ = self.forward(X)
        if self.config.loss == "cross_entropy":
            return cross_entropy_loss(y, A_out)
        else:
            return mean_squared_error(y, A_out)
    
    def _compute_accuracy(self, X, y):
        A_out, _ = self.forward(X)
        predictions = np.argmax(A_out, axis=0)
        return np.mean(predictions == np.argmax(y, axis=0))

def main(config):
    wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=config)
    
    if config.dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0
    y_train = np.eye(10)[y_train].T
    y_test = np.eye(10)[y_test].T
    
    X_train, X_val = X_train[:, :50000], X_train[:, 50000:]
    y_train, y_val = y_train[:, :50000], y_train[:, 50000:]
    
    model = NeuralNetwork(config)
    model.train(X_train, y_train, X_val, y_val)
    
    test_accuracy = model._compute_accuracy(X_test, y_test)
    wandb.log({"test_accuracy": test_accuracy})
    print(f"Test Accuracy: {test_accuracy:.4f}")

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
