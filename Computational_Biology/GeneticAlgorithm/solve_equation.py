import GA
import random
import math

MAX_GEN = 100

class Solver:
    def __init__(self):
        pop = self.create_population()
        self.GA = GA.GA(pop,
            self.end_condition,
            self.fitness,
            crossover_probability=0.3,
            mutation_probability=0.2
        )
        self.last_max_val = None

    def run(self):
        self.GA.run()
        max_val, x = self.evaluate_population(self.GA.population)
        print('x = {}, max_val = {}'.format(x, max_val))

    def extract_x(self, offspring):
        offspring = ''.join(offspring)
        x = int(offspring, base=2)
        return x

    def fitness(self, offspring):
        x = self.extract_x(offspring)
        return 31*x - x*x

    def post_build_event(self, pop):
        fitnesses = [self.fitness(x) for x in pop]
        total_fitness = sum(fitnesses)
        #set the threshold to be avg. of fitnesses
        threshold = total_fitness / len(pop)
        new_pop = []
        for offspring,fitness in zip(pop, fitnesses):
            if fitness >= threshold:
                new_pop.append(offspring)
        return new_pop

    def evaluate_population(self, pop):
        max_val = None
        max_x = None
        for offspring in pop:
            val = self.fitness(offspring)
            if max_val is None or val > max_val:
                max_val = val
                max_x = self.extract_x(offspring)
        return max_val, max_x

    def end_condition(self, pop, gen_id):
        if gen_id >= MAX_GEN:
            return True
        cur_max_val = self.evaluate_population(pop)
        if self.last_max_val is None:
            self.last_max_val = cur_max_val
        elif cur_max_val == self.last_max_val:
            return True
        return False

    def create_population(self):
        pop = []
        for i in range(100):
            pop.append(list('{:05b}'.format(random.randint(0, 31))))
        return pop

