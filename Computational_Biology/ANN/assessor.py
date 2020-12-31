from z3 import *

class Assessor:
    def __init__(self, inputs_num):
        self.solver = Solver()
        self.inputs_num = inputs_num
        self.weights = [Real('w{}'.format(x)) for x in range(self.inputs_num)]
        self.threshold = Real('threshold')

    def assess_sample(self, sample, val):
        expr = None
        for w, x in zip(self.weights, sample):
            subexpr = w*x
            if expr is None:
                expr = subexpr
            else:
                expr = expr.__add__(subexpr)
        if val == 1:
            expr = expr.__ge__(self.threshold)
        else:
            expr = expr.__lt__(self.threshold)
        self.solver.add(expr)
        return self.solver.check() == sat
        
    def assess(self, samples, values):
        result = True
        for x,y in zip(samples, values):
            result = self.assess_sample(x, y)
            if not result:
                break
        return result

def is_xor_solvable():
    d = Assessor(2)
    X = [[0,0],
        [0, 1],
        [1, 0],
        [1, 1]]
    Y = [0, 1, 1, 0]
    return d.assess(X, Y)
    
if __name__ == '__main__':
    print('xor is solvable: {}'.format(is_xor_solvable()))
