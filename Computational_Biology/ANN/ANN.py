import logging

class Neuron:
    def __init__(self, threshold):
        self.threshold = threshold
        self.inputs = []

    def connect(self, output, wheight):
        self.output.add_input(self, wheight)

    def add_input(self, input_neuron, wheight):
        self.inputs.append([input_neuron, wheight])

    def add_inputs(self, *inputs_and_wheights):
        """
        @param inputs_and_wheights: list of tuple of (input_neuron, wheight)
        """
        for input_neuron, wheight in inputs_and_wheights:
            self.add_input(input_neuron, wheight)

    def activate(self):
        result = 0
        for neuron, wheight in self.inputs:
            result += neuron.activate() * wheight
        if result >= self.threshold:
            return 1
        return -1

class InputNeuron(Neuron):
    """
    A special neuron for input
    """
    def __init__(self, value):
        super().__init__(threshold=None)
        self.value = value

    def activate(self):
        return self.value

def test_majority(x1, x2, x3):
    n1 = InputNeuron(x1)
    n2 = InputNeuron(x2)
    n3 = InputNeuron(x3)
    majority_neuron = Neuron(1)
    majority_neuron.add_inputs(
            (n1, 0.8),
            (n2, 0.7),
            (n3, 0.6)
    )
    result = majority_neuron.activate()
    return int(result > 0)

class Perceptron:
    def __init__(self, threshold=0, log_level=logging.INFO):
        self.threshold = threshold
        self.output = None
        logging.basicConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

    def _set_inputs(self, x):
        for idx, inp in enumerate(self.output.inputs):
            inp[0].value = x[idx]

    def _get_wheights(self):
        return [x[1] for x in self.output.inputs]

    def fit(self, X, Y, max_iterations=None):
        #init the network
        if len(X) == 0:
            raise Exception("zero samples given")
        n = len(X[0])
        inputs = [InputNeuron(0) for _ in range(n)]
        wheights = [0 for _ in range(n)]
        self.output = Neuron(threshold=self.threshold)
        self.output.add_inputs(*zip(inputs, wheights))
        #learn
        k = 0
        is_learning_succeeded = False
        while max_iterations is None or (k < max_iterations):
            self.logger.info("iteration %d" % (k))
            is_perfect_wheighted = True
            for x,y in zip(X,Y):
                #init the inputs to be x
                self._set_inputs(x)
                #activate the network
                x_class = self.output.activate()
                if x_class != y:
                    is_perfect_wheighted = False

                    #the classification is incorrect
                    if y == 1:
                        sign = 1
                    elif y == -1:
                        sign = -1
                    else:
                        raise Exception("class is invalid (%s)" % (y))

                    self.logger.debug("class incorrect (%s instead of %s)" % (x_class, y))
                    self.logger.debug("current wheights: %s" % (self._get_wheights()))
                    for idx, inp in enumerate(self.output.inputs):
                        inp[1] += (x[idx] * sign)
                    self.logger.debug("updated to new wheights: %s" % (self._get_wheights()))
            if is_perfect_wheighted:
                is_learning_succeeded = True
                self.logger.debug("Learning done!")
                break

            k += 1

        if is_learning_succeeded:
            self.logger.info("Learning succeeded!")
        else:
            self.logger.info("Learning failed")

    def predict(self, x):
        self._set_inputs(x)
        return self.output.activate()
