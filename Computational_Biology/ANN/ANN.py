import logging
from collections import Iterable, defaultdict
import numpy as np
from math import e

class ActivationFunc:
    def __call__(self, x):
        raise NotImplementedError()

    def get_derivative(self, x):
        raise NotImplementedError()

class Tanh(ActivationFunc):
    def __call__(self, x):
        return (e**x - e**(-x))/(e**x + e**(-x))

    def get_derivative(self, x):
        #according to wolfarm alpha
        return 4/((e**(-x) + e**x)**2)

class Sigmoid(ActivationFunc):
    def __call__(self, x):
        return 1/(1 + e**(-x))

    def get_derivative(self, x):
        return e**x/((e**x + 1)**2)

class Neuron:
    def __init__(self, threshold, activation_func=None):
        self.threshold = threshold
        self.inputs = []
        #float32 is enough
        self.weights = np.array([], dtype=np.float32)
        #last value when this neuron was activated
        self.last_val = None
        #last value without the threshold or the activation_func
        self.last_val_h = None
        self.activation_func = activation_func

    def connect(self, output, wheight):
        self.output.add_input(self, wheight)

    def add_input(self, input_neuron, wheight):
        """
        Adds an input neuron
        @returns: input_index
        """
        input_index = len(self.inputs)
        self.inputs.append(input_neuron)
        self.weights = np.append(self.weights, wheight)
        return input_index

    def add_inputs(self, *inputs_and_wheights):
        """
        @param inputs_and_wheights: list of tuple of (input_neuron, wheight)
        """
        for input_neuron, wheight in inputs_and_wheights:
            self.add_input(input_neuron, wheight)

    def _activate_input(self, neuron):
        if isinstance(neuron, Neuron):
            if neuron.last_val is not None:
                return neuron.last_val
            return neuron.activate()
        else:
            #its a basic type - an input or bias
            return neuron

    def activate(self):
        inputs = np.array([self._activate_input(n) for n in self.inputs])
        result = np.dot(inputs, self.weights)
        self.last_val_h = result
        if self.activation_func:
            result = self.activation_func(result)
        else:
            if result >= self.threshold:
                result = 1
            else:
                result = 0 #-1
        self.last_val = result
        return self.last_val

def test_majority(x1, x2, x3):
    n1 = x1
    n2 = x2
    n3 = x3
    majority_neuron = Neuron(1)
    majority_neuron.add_inputs(
            (n1, 0.8),
            (n2, 0.7),
            (n3, 0.6)
    )
    result = majority_neuron.activate()
    return int(result > 0)

def test_lot_inputs():
    import random
    inputs = [random.randint(0, 100) for x in range(100*100)]
    weights = [random.randint(0, 100) / 100 for x in range(100*100)]
    output = Neuron(1)
    output.add_inputs(*zip(inputs, weights))
    result = output.activate()
    return int(result > 0)

class Perceptron:
    def __init__(self, threshold=0, log_level=logging.INFO):
        self.threshold = threshold
        self.output = None
        logging.basicConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

    def _set_inputs(self, x):
        self.output.inputs = x

    def _get_weights(self):
        return self.output.weights

    def fit(self, X, Y, max_iterations=None, learning_rate=1):
        #init the network
        if len(X) == 0:
            raise Exception("zero samples given")
        n = len(X[0])
        inputs = [0 for _ in range(n)]
        wheights = [0 for _ in range(n)]
        self.output = Neuron(threshold=self.threshold)
        self.output.add_inputs(*zip(inputs, wheights))
        #learn
        k = 0
        is_learning_succeeded = False
        while max_iterations is None or (k < max_iterations):
            self.logger.info("iteration %d" % (k))
            is_perfect_wheighted = True
            number_of_fixes = 0
            for x,y in zip(X,Y):
                #init the inputs to be x
                self._set_inputs(x)
                #activate the network
                x_class = self.output.activate()
                if x_class != y:
                    is_perfect_wheighted = False
                    number_of_fixes += 1

                    #the classification is incorrect
                    if y == 1:
                        sign = 1
                    elif y == -1:
                        sign = -1
                    else:
                        raise Exception("class is invalid (%s)" % (y))

                    self.logger.debug("class incorrect (%s instead of %s)" % (x_class, y))
                    self.logger.debug("current wheights: %s" % (self._get_weights()))
                    for idx, _ in enumerate(self.output.weights):
                        self.output.weights[idx] += (x[idx] * sign * learning_rate)
                    self.logger.debug("updated to new wheights: %s" % (self._get_weights()))
            if is_perfect_wheighted:
                is_learning_succeeded = True
                self.logger.debug("Learning done!")
                break

            success_ratio = (len(X) - number_of_fixes)/len(X) * 100
            self.logger.info(f"number of fixes: {number_of_fixes}, success ratio = {success_ratio}")
            k += 1

        if is_learning_succeeded:
            self.logger.info("Learning succeeded!")
        else:
            self.logger.info("Learning failed")

    def predict(self, x):
        self._set_inputs(x)
        return self.output.activate()

class NetworkArch:
    def __init__(self, inputs_count, outputs_count, *inners_count):
        """
        @param inputs_count: the number of inputs neurons
        @param outputs_count: the number of outputs neurons
        @param inners_count: a list of counts of each inner layer, by order
        """
        self.layers_counts = [inputs_count] + list(inners_count) + [outputs_count]
        #a dict of (n1_index, n2_index) -> weight
        self.edges = {}
        #a dict of n1_index -> n2_index
        self.src_to_dest = defaultdict(list)

    def connect(self, n1_index, n2_index, weight):
        """
        @param n1_index: a tuple of (layer_id, neuron_id)
        @param n2_index: a tuple of (layer_id, neuron_id)
        """
        if (n1_index, n2_index) not in self.edges:
            self.edges[(n1_index, n2_index)] = weight
            self.src_to_dest[n1_index].append(n2_index)

    def connect_layer(self, layer_id, neuron_index, *weight_or_weigths):
        """
        connects a whole layer to a signle neuron
        @param layer_id: the input layer id
        @param neuron_index: the output neuron index
        @param weight_or_weights: a weight for each edge or a list of weights for each edge
        """
        weights = None
        if isinstance(weight_or_weigths, Iterable):
            weights = weight_or_weigths
        else:
            weights = [weight_or_weigths] * self.layers_counts[layer_id]
        for i in range(self.layers_counts[layer_id]):
            self.connect((layer_id, i), neuron_index, weights[i])


class BackpropNetwork:
    def __init__(self, network_arch, threshold=0, activation_func=None, log_level=logging.INFO):
        self.threshold = threshold
        self.output = None
        logging.basicConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self.network_arch = network_arch
        #the layers in self.network starts after the input layer
        self.network = []
        #neurons that has an input that comes from the first layer of the inputs
        #a list of tuples of (neuron, input_index)
        self.second_layer_neurons = []
        #add the inputs layer
        self.network.append(np.array([0] * self.network_arch.layers_counts[0]))
        for layer_counts in self.network_arch.layers_counts[1:]:
            self.network.append([])
            for i in range(layer_counts):
                n = Neuron(threshold=threshold, activation_func=activation_func)
                self.network[-1].append(n)

        #connect the neurons according to the edges given by self.network_arch
        for src_dst, weight in self.network_arch.edges.items():
            src, dst = src_dst
            src_n = self.network[src[0]][src[1]]
            dst_n = self.network[dst[0]][dst[1]]
            #add input to dst_n
            input_index = dst_n.add_input(src_n, weight)
            if src[0] == 0:
                #its a first layer neuron - add the dst neuron as a second layer
                #this applies also for CNN - any neuron that has an input neuron from the first layer
                #will be called here a second-layer-neuron
                self.second_layer_neurons.append((dst_n, input_index))

    def _set_inputs(self, x):
        self.network[0] = np.array(x)
        #set the inputs of the first layer
        for dst, input_index in self.second_layer_neurons:
            #input_index is also an index in x, since x is a sample for the full input layer
            dst.inputs[input_index] = x[input_index]

    def _clear_last_val(self):
        for layer_id in range(1, len(self.network)):
            for n in self.network[layer_id]:
                n.last_val = None
                n.last_val_h = None

    def backprop(self, y, learning_rate=1):
        #deltas is a dict of (layer_id, neuron_id) -> delta
        deltas = {}
        #walk from the last layer to the second layer
        for layer_index in range(len(self.network) - 1, 0, -1):
            for index, neuron in enumerate(self.network[layer_index]):
                neuron_index = (layer_index, index)
                activation_func = neuron.activation_func
                if activation_func is None:
                    #if its None then the activation func is lambda x: x
                    #so the derivative is 1
                    g_d = 1
                else:
                    g_d = neuron.activation_func.get_derivative(neuron.last_val_h)
                sum_signals = 0
                if layer_index == len(self.network) - 1:
                    #its the output layer
                    sum_signals = y[index] - neuron.last_val
                else:
                    for n2_index in self.network_arch.src_to_dest[(layer_index, index)]:
                        weight = self.network_arch.edges[(neuron_index, n2_index)]
                        sum_signals += deltas[n2_index] * weight
                delta = g_d * sum_signals
                deltas[neuron_index] = delta

        #update weights
        for layer_id in range(len(self.network)):
            for index in range(self.network_arch.layers_counts[layer_id]):
                n_index = (layer_id, index)
                src = self.network[layer_id][index]
                #the source neuron value
                S = None
                if layer_id == 0:
                    S = src
                else:
                    S = src.last_val
                for dst_neuron in self.network_arch.src_to_dest[n_index]:
                    old_weight = self.network_arch.edges[(n_index, dst_neuron)]
                    #the error signal of dest neuron
                    delta = deltas[dst_neuron]
                    new_weight = learning_rate * S * delta
                    #update in network_arch
                    self.network_arch.edges[(n_index, dst_neuron)] = new_weight
                    #update in self.network
                    dst = self.network[dst_neuron[0]][dst_neuron[1]]
                    #find according to object id, should be fast enough
                    weight_index = dst.inputs.index(src)
                    dst.weights[weight_index] = new_weight

    def fit(self, X, Y, max_iterations=None, learning_rate=1):
        #init the network
        if len(X) == 0:
            raise Exception("zero samples given")
        n = len(X[0])
        if n != self.network_arch.layers_counts[0]:
            raise Exception("invalid shape of sample (got {} should be {})".format(
                n, self.network_arch.layers_counts[0]))
        if not all(n == len(smpl) for smpl in X):
            raise Exception("not all of the inputs has the same shape")
        for x,y in zip(X, Y):
            import ipdb; ipdb.set_trace()
            self._clear_last_val()
            self._set_inputs(x)
            #start with calculating the result of the outputs neurons
            for output in self.network[-1]:
                output.activate()
            #run backprop algorithm
            self.backprop(y, learning_rate=learning_rate)
        

def test_backprop():
    net_arch = NetworkArch(3, 1, 3)
    net_arch.connect((0, 0), (1, 0), -0.1) #w1
    net_arch.connect((0, 0), (1, 1), 0) #w2

    net_arch.connect((0, 1), (1, 0), 0.1) #w3
    net_arch.connect((0, 1), (1, 1), 0.2) #w4

    net_arch.connect((0, 2), (1, 0), 0.1) #w5
    net_arch.connect((0, 2), (1, 1), -0.2) #w6

    net_arch.connect((1, 0), (2, 0), 0.1) #w7
    net_arch.connect((1, 1), (2, 0), -0.3) #w8
    net_arch.connect((1, 2), (2, 0), 0) #w9

    sigmoid = Sigmoid()
    net = BackpropNetwork(net_arch, activation_func=sigmoid)
    X = [(0,1, -1), (1,1, -1)]
    Y = [(1,), (0,)]
    net.fit(X, Y, learning_rate=0.5)
