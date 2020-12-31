import random
from copy import copy

class GA:
    def __init__(
            self,
            population,
            end_condition_func,
            fitness_func,
            crossover_probability,
            mutation_probability,
            crossover_points=1,
            select=None,
            post_build_event=None,
            mutate_func=None,
            crossover_func=None
        ):
        """
        @param popultion: list of popultion to start with
        @param end_condition_func: a function that tells if we got our desired solution
            this function takes population and generation number as a parameters
        @param fitness_func: a function that calculates the fitness of the population
            takes an individual as a parameter
        """
        assert len(population) > 0, 'population cannot be empty'

        self.population = population
        self.end_condition = end_condition_func
        self.fitness_func = fitness_func
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.crossover_points = crossover_points
        if select is None:
            self.select = self.roulette_wheel_select
        else:
            self.select = {
                    'roulette_wheel':self.roulette_wheel_select,
                    'rank':self.rank_select,
                    'tournament':self.tournament_select
            }[select]
        if mutate_func is None:
            mutate_func = self.mutate
        if crossover_func is None:
            crossover_func = self.crossover
        self.mutate_func = mutate_func
        self.crossover_func = crossover_func
        self.post_build_event = post_build_event
        self.tournament_competetors = 2


        self.init_pop_size = len(self.population)
        self.generation_id = 0

        self.current_fitnesses = None
        self.total_fitness = None

    def run(self):
        while not self.end_condition(self.population, self.generation_id):
            self.population = self.next_generation()
            if self.post_build_event:
                self.population = self.post_build_event(self.population)
            self.generation_id += 1

    def next_generation(self):
        new_pop = []
        self.current_fitnesses = [self.fitness_func(x) for x in self.population]
        self.total_fitness = sum(self.current_fitnesses)
        print('Generation {}, fitness: max = {}, min = {}, avg. = {}, total_fitness = {}'.format(
            self.generation_id,
            max(self.current_fitnesses),
            min(self.current_fitnesses),
            self.total_fitness / len(self.current_fitnesses),
            self.total_fitness
        ))
        while not self.is_full(new_pop):
            parent1 = self.select()
            parent2 = self.select()
            offspring = self.crossover_func(parent1, parent2, self.crossover_probability)
            offsprint = self.mutate_func(offspring, self.mutation_probability)
            new_pop.append(offspring)
        return new_pop

    def is_full(self, population):
        if len(population) >= self.init_pop_size:
            return True
        return False

    def crossover(self, parent1, parent2, pc):
        resulting_chromosomes = []
        if random.random() < pc:
            points = []
            for i in range(self.crossover_points):
                #according to the documentation randint includes both end points
                point = random.randint(0, len(parent1) - 1)
                points.append(point)
            points = sorted(points)
            points.append(len(parent1))
            o1 = None
            o2 = None
            parents = [parent1, parent2]
            cur_parent = 0
            next_parent = lambda x: (x + 1) % len(parents)
            for p_idx in range(len(points)):
                p = points[p_idx]
                last_p = points[p_idx - 1] if p_idx > 0 else 0
                sect1 = parents[cur_parent][last_p:p]
                sect2 = parents[next_parent(cur_parent)][last_p:p]
                if o1 is None:
                    o1 = sect1
                    o2 = sect2
                else:
                    o1 += sect1
                    o2 += sect2
                cur_parent = next_parent(cur_parent)
            resulting_chromosomes = [
                    parent1[:point] + parent2[point:],
                    parent2[:point] + parent1[point:]
            ]
        else:
            resulting_chromosomes += [parent1, parent2]
        return random.choice(resulting_chromosomes)

    def mutate(self, offspring, pm):
        replacement = {'1': '0', '0': '1'}
        for k in range(len(offspring)):
            if random.random() < pm:
                offspring[k] = replacement[offspring[k]]
        return offspring

    def roulette_wheel_select(self):
        wheel_location = random.random() * self.total_fitness
        i = 0
        current_sum = 0
        while current_sum < wheel_location and i < len(self.population) - 1:
            current_sum += self.current_fitnesses[i]
            #increment
            i += 1
        return self.population[i]

    def rank_select(self):
        max_fitness = None
        max_idx = None
        for i in range(len(self.population)):
            cur_fitness = self.current_fitnesses[i]
            if max_fitness is None or cur_fitness > max_fitness:
                max_fitness = cur_fitness
                max_idx = i
        return self.population[max_idx]

    def tournament_select(self):
        competetors = []
        for i in range(self.tournament_competetors):
            for j in range(3):
                c = random.choice(range(0, len(self.population)))
                if c not in competetors:
                    competetors.append(c)
                    break
        max_fitness, max_idx = None, None
        for c in competetors:
            if max_fitness is None or self.current_fitnesses[c] > max_fitness:
                max_fitness = self.current_fitnesses[c]
                max_idx = c
        return self.population[max_idx]
