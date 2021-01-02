import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length
    return ((y_true - y_pred) ** 2).mean()

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1-fx)

class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
    def feedforward(self, inputs):
        total = np.dot(self.weight, inputs) + self.bias
        return sigmoid(total)

weights = np.array([0,1])
bias = 4
n = Neuron(weights, bias)

class OurNeuralNetwork:
    '''
    A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)

    '''
    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
          Elements in all_y_trues correspond to those in data.
        '''
        learn_rate = 0.1
        epochs = 1000 # number of times to loop through the entire dataset

        for epoch in range(epochs):
          for x, y_true in zip(data, all_y_trues):
            
            sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
            h1 = sigmoid(sum_h1)

            sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
            h2 = sigmoid(sum_h2)

            sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
            o1 = sigmoid(sum_o1)
            y_pred = o1

            # Calculate partial derivatives.
            dL_dy_pred = -2 * (y_true - y_pred)

            # Neuron o1
            dy_pred_dw5 = h1 * deriv_sigmoid(sum_o1)
            dy_pred_dw6 = h2 * deriv_sigmoid(sum_o1)
            dy_pred_db3 = deriv_sigmoid(sum_o1)

            dy_pred_dh1 = self.w5 * deriv_sigmoid(sum_o1)
            dy_pred_dh2 = self.w6 * deriv_sigmoid(sum_o1)

            # Neuron h1
            dh1_dw1 = x[0] * deriv_sigmoid(sum_h1)
            dh1_dw2 = x[1] * deriv_sigmoid(sum_h1)
            dh1_db1 = deriv_sigmoid(sum_h1)

            # Neuron h2
            dh2_dw3 = x[0] * deriv_sigmoid(sum_h2)
            dh2_dw4 = x[1] * deriv_sigmoid(sum_h2)
            dh2_db2 = deriv_sigmoid(sum_h2)

            # Update weights and biases
            # Neuron h1
            self.w1 -= learn_rate * dL_dy_pred * dy_pred_dh1 * dh1_dw1
            self.w2 -= learn_rate * dL_dy_pred * dy_pred_dh1 * dh1_dw2
            self.b1 -= learn_rate * dL_dy_pred * dy_pred_dh1 * dh1_db1
            
            # Neuron h2
            self.w3 -= learn_rate * dL_dy_pred * dy_pred_dh2 * dh2_dw3
            self.w4 -= learn_rate * dL_dy_pred * dy_pred_dh2 * dh2_dw4
            self.b2 -= learn_rate * dL_dy_pred * dy_pred_dh2 * dh2_db2

            # Neuron o1
            self.w5 -= learn_rate * dL_dy_pred * dy_pred_dw5
            self.w6 -= learn_rate * dL_dy_pred * dy_pred_dw6
            self.b3 -= learn_rate * dL_dy_pred * dy_pred_db3

          # --- Calculate total loss at the end of each epoch
          if epoch % 10 == 0:
            y_preds = np.apply_along_axis(self.feedforward, 1, data)
            loss = mse_loss(all_y_trues, y_preds)
            print("Epoch %d loss: %.3f" % (epoch, loss))

        
# Define dataset
data = np.array([
  [-10, -2],  # Alice
  [19, -5],   # Bob
  [11, -3],   # Charlie
  [-21, -7], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)

# Make some predictions
emily = np.array([-12, -4]) # 128 pounds, 63 inches
frank = np.array([15, 1])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M
