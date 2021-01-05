import math
import random
import re
from collections import OrderedDict

from z3 import *

# a rounding of math.e
E = 2.7183

class Assessor:
    def __init__(self, inputs_num):
        self.init_solver()
        self.inputs_num = inputs_num
        self.weights = [Real('w{}'.format(x)) for x in range(self.inputs_num)]
        self.threshold = Real('threshold')

    def init_solver(self):
        self.solver = Solver()

    def get_expr(self, sample, val):
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
        return expr

    def get_exprs(self, samples, values):
        exprs = []
        for x,y in zip(samples, values):
            exprs.append(self.get_expr(x, y))
        return exprs

    def assess_sample(self, sample, val):
        expr = self.get_expr(sample, val)
        self.solver.add(expr)
        res = self.solver.check()
        if res == sat:
            return True
        elif res == unsat:
            return False
        elif res == unknown:
            raise Exception('got `unknown` result from the solver')
        else:
            raise Exception('got res = {}'.format(res))
        
    def assess(self, samples, values):
        #reinitialize the solver
        self.init_solver()
        result = True
        for x,y in zip(samples, values):
            result = self.assess_sample(x, y)
            if not result:
                break
        return result

class ComplexAssersor(Assessor):
    def __init__(self, expr_text):
        self.init_solver()
        self.expr_text = expr_text
        self.vars = {}
        self._parse_expr(self.expr_text)

    def get_symbolic_var(self, name):
        name += str(len(self.vars.keys()))
        var = Real(name)
        self.vars[name] = var
        return var

    def gen_weight(self):
        name = 'w'
        return self.get_symbolic_var(name)

    def get_expr(self, sample, value):
        expr = self.expr
        if not any(x in str(expr.decl()) for x in ('>', '<')):
            #assume that the threshold is of all network
            if value == 1:
                expr = expr.__ge__(self.threshold)
            else:
                expr = expr.__lt__(self.threshold)
        #replace symbolic vars in values
        subs = []
        for idx,v in enumerate(self.variables_syms.values()):
            subs.append((v, RealVal(sample[idx])))
        expr = substitute(expr, *subs)
        return expr

    def sigmoid(self, expr):
        #expr = 1 / (1 + RealVal(E) ** (-expr))

        #taylor expansion
        expr = (1/2) + expr/4 - (expr**3)/48
        return expr

    def tanh(self, expr):
        #e_x = RealVal(E) ** (expr)
        #e_minus_x = RealVal(E) ** (-expr)
        #expr = (e_x - e_minus_x) / (e_x + e_minus_x)

        #taylor expansion
        expr = expr - (expr**3)/3
        return expr

    def ReLU(self, expr):
        expr = If(expr <= 0, 0, expr)
        return expr

    def _parse_expr(self, expr_text):
        weights = re.findall('w[0-9]+', expr_text)
        variables = re.findall('x[0-9]+', expr_text)
        thresholds = re.findall('t[0-9]+', expr_text)
        self.weights_syms = {}
        self.variables_syms = OrderedDict({})
        self.thresholds_syms = {}
        for w in weights:
            if w not in self.weights_syms:
                self.weights_syms[w] = self.gen_weight()
            expr_text = expr_text.replace(w, 'self.weights_syms["{}"]'.format(w))
        for idx,v in enumerate(variables):
            if v not in self.variables_syms:
                self.variables_syms[v] = self.get_symbolic_var('x')
            expr_text = expr_text.replace(v, 'self.variables_syms["{}"]'.format(v))
        for t in thresholds:
            if t not in self.thresholds_syms:
                self.thresholds_syms[t] = self.get_symbolic_var('t')
            expr_text = expr_text.replace(t, 'self.thresholds_syms["{}"]'.format(t))

        activation_funcs = ['sigmoid', 'tanh', 'ReLU']
        for af in activation_funcs:
            expr_text = expr_text.replace(af, 'self.{}'.format(af))

        # eval, later we can use pyparsing
        self.threshold = self.get_symbolic_var('t_output')
        self.expr = eval(expr_text)


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
