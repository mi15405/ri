import sys
import itertools as itool
from graphics import *
import random
import numpy as np

class Genetic():
    def __init__(
            self, 
            population = [], 
            mutation_chance = 0.5, 
            mutation_range = 0.1, 
            crossover_cuts = 0.1,
            elite = 0.05, 
            genocid = 0.05):
        self.population = population
        self.mut_chance = mutation_chance
        self.mut_range = mutation_range
        self.cross_cuts = len(population) * crossover_cuts
        self.elite = elite
        self.genocid = genocid
        self.max_iter = 100
        self.visual_step = 10
        self.is_animating = False
        random.seed()

    def set_window(self, window):
        self.win = window

    def run_simulation(self, with_visual):
        if with_visual:
            # Draw movers
            for mover in self.population:
                mover.draw(self.win)
            self.is_animating = True
        # Simulate one generation
        self.simulate();

        if with_visual:
            # Undraw movers
            for mover in self.population:
                mover.undraw()
            self.is_animating = False

    def start_simulation(self):
        print('Simulation started!')  

        for it in range(self.max_iter):
            print('ITERATION: %d' % it)
            
            # Show every 'visual step' iteration
            self.run_simulation(it % self.visual_step == 0)
                
            # Calculate and desc order fitness: (index, fitness)
            fitness = self.order_by_fitness()

            # Population size
            pop_size = len(self.population)

            # Population elite size
            elite_size = int(pop_size * self.elite)

            # elite size is always even
            if elite_size % 2 == 1:
                elite_size -= 1

            # Save elite indexes
            elite_index = fitness[:elite_size, 0]

            # Initialize new population with elite
            new_population = list(self.population[int(i)] for i in elite_index)

            # Population genocid size
            genocid_size = int(pop_size * self.genocid)

            # Reduce initial population
            if genocid_size != 0:
                #self.population = self.population[:-genocid_size]
                fitness = fitness[:-genocid_size]

            # Add new generation to new population
            new_population += self.reproduce(fitness, pop_size - elite_size)
            self.population = new_population

            # Reset positions and velocities
            for mover in self.population:
                mover.reset()

            # Next iteration
            it += 1

    def reproduce(self, fitness, size):
        # Accumulated values
        acc = itool.accumulate(list(fitness[:,1]))

        # Set accumulated fitness
        for i, newValue in enumerate(acc):
            fitness[i, 1] = newValue 

        newGeneration = []
        for i in range(size//2):
            parents = self.select_parents(fitness)
            child1, child2 = self.crossover(parents)

            for child in child1, child2:
                child.mutate(self.mut_chance, self.mut_range)
                newGeneration.append(child)
        return newGeneration

    def select_parents(self, fitness):
        parent1 = int(Genetic.select_from_pool(fitness, random.uniform(0, 1)))
        parent2 = int(Genetic.select_from_pool(fitness, random.uniform(0, 1)))
        return self.population[parent1], self.population[parent2]

    @staticmethod
    def select_from_pool(pool, selection):
        for x in pool:
            if x[1] >= selection:
                return x[0]
        return pool[-1][0]


    def crossover(self, parents):
        child1, child2 = parents[0].clone(), parents[1].clone()
        cutPoint = int(np.size(child1.chromosome, 0) / 2)
        
        # Number of segments --> number of cuts + 1
        segment_num = self.cross_cuts + 1
        chrom_size = np.size(child1.chromosome, 0)

        cross = True
        for i in range(chrom_size):
            gene1, gene2 = parents[0].chromosome[i], parents[1].chromosome[i]
            if i % segment_num == 0:
                cross = not cross
            if cross:
                child1.chromosome[i] = gene2
                child2.chromosome[i] = gene1
            else:
                child1.chromosome[i] = gene1
                child2.chromosome[i] = gene2

        return child1, child2

    def set_environment(self, field, obstacles, goal):
        self.field = field
        self.obstacles = obstacles
        self.goal = goal

    def order_by_fitness(self): 
        fitness = np.array(
                [np.array([i, x.fitness()]) for i, x in enumerate(self.population)])

        sum_fitness = sum(fit for index, fit in fitness)

        # Normalized fitness
        fitness[:,1] /= sum_fitness

        # Sorted fitness
        fitness = np.array(sorted(fitness, key = lambda x: x[1], reverse = True))
        return fitness


    def simulate(self):
        while any(mover.alive for mover in self.population):
            for mover in self.population:
                mover.update()

                in_field = mover.isInside(self.field)
                touched_obstacle = any(mover.isInside(obst) for obst in self.obstacles)
                reached_goal = mover.isInside(self.goal)

                if not in_field or touched_obstacle:
                    mover.kill()
                elif reached_goal:
                    mover.reachedGoal = True
                    mover.kill()

            if self.is_animating:
                update(30)
