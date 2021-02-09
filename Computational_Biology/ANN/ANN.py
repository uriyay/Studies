import logging
import numpy as np

class Neuron:
    def __init__(self, threshold):
        self.threshold = threshold
        self.inputs = []
        #float32 is enough
        self.weights = np.array([], dtype=np.float32)

    def connect(self, output, wheight):
        self.output.add_input(self, wheight)

    def add_input(self, input_neuron, wheight):
        self.inputs.append(input_neuron)
        self.weights = np.append(self.weights, wheight)

    def add_inputs(self, *inputs_and_wheights):
        """
        @param inputs_and_wheights: list of tuple of (input_neuron, wheight)
        """
        for input_neuron, wheight in inputs_and_wheights:
            self.add_input(input_neuron, wheight)

    def _activate_input(self, neuron):
        if isinstance(neuron, Neuron):
            return neuron.activate()
        else:
            #its a basic type - an input or bias
            return neuron

    def activate(self):
        inputs = np.array([self._activate_input(n) for n in self.inputs])
        result = np.dot(inputs, self.weights)
        if result >= self.threshold:
            return 1
        return -1

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
