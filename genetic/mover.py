import sys
import itertools as itool
from graphics import *
import random
import numpy as np
import time

class Mover():
    def __init__(self, x, y, speed, radius, lifetime, numOfGenes, goal):
        self.x = x
        self.y = y
        self.start = np.array([x, y])
        self.position = np.array([x, y])
        self.alive = True
        self.life = lifetime
        self.speed = speed
        self.circle = Circle(Point(x, y), radius)
        self.circle.setFill('black')
        self.is_drawn = False
        self.goal = goal
        self.lifetime = lifetime
        self.numOfGenes = numOfGenes
        self.chromosome = Mover.random_vectors(numOfGenes, -speed, speed)
        self.reset_vel()
        self.acc = self.chromosome[0]
        self.accIndex = 0
        self.reachedGoal = False

    def reset_vel(self):
        self.vel = np.array([0., 0.]) 

    def clone(self):
        return Mover(self.start[0], self.start[1], self.speed,
                self.circle.getRadius(), self.lifetime, self.numOfGenes, self.goal)

    def update(self):
        if not self.alive:
            return
        else: 
            self.life -= 1
            if self.life < 0:
                self.kill()

        self.vel += self.acc
        self.position += self.vel
        self.circle.move(self.vel[0], self.vel[1])

        self.accIndex = (self.accIndex + 1) % len(self.chromosome)
        self.acc = self.chromosome[self.accIndex]

    def fitness(self):
        fit = 1.0 / np.linalg.norm(self.goal - self.position)
        if self.reachedGoal:
            fit += self.life
        return fit

    def reset(self):
        toStart = self.start - self.position
        self.circle.move(toStart[0], toStart[1])
        self.position = np.array(self.start)
        self.life = self.lifetime
        self.alive = True
        self.vel = np.array([0., 0.])
        self.acc = self.chromosome[0]
        self.reachedGoal = False
        self.accIndex = 0

    def draw(self, win):
        if not self.is_drawn:
            self.circle.draw(win)
            self.is_drawn = True

    def undraw(self):
        if self.is_drawn:
            self.circle.undraw()
            self.is_drawn = False

    def mutate(self, mut_chance, mut_range):
        # random broj
        if random.uniform(0, 1) <= mut_chance:
            return
        # Number of genes in chromosome
        size = np.size(self.chromosome, 0)

        # Number of genes to mutate
        mut = int(mut_range * size)

        # First to mutate
        start = random.randint(0, size -mut -1)

        for i in range(mut):        
            self.chromosome[start + i][0] = random.uniform(-self.speed, self.speed)
            self.chromosome[start + i][1] = random.uniform(-self.speed, self.speed)

    @staticmethod
    def random_vectors(length, minValue, maxValue):
        vectors = np.repeat([[0., 0.]], length, axis = 0)
        for i in range(length):
            x = random.uniform(minValue, maxValue)
            y = random.uniform(minValue, maxValue)
            vectors[i][0] = x
            vectors[i][1] = y
        return vectors
    
    def kill(self):
        self.alive = False

    def isInside(self, rect):
        center = self.circle.getCenter()
        p1, p2 = rect.getP1(), rect.getP2()

        if center.x >= p1.x and center.x <= p2.x and\
                center.y >= p1.y and center.y <= p2.y:
            return True
        else:
            return False

def randomInts(num, left, right):
    numbers = []
    for i in range(num):
        numbers.append(random.randint(left, right))

    return numbers

class Genetic():
    def __init__(
            self, 
            population = [], 
            mutation_chance = 0.2, 
            mutation_range = 0.1, 
            elite = 0.1, 
            genocid = 0.5):
        self.population = population
        self.mut_chance = mutation_chance
        self.mut_range = mutation_range
        self.elite = elite
        self.genocid = genocid
        self.max_iter = 100
        self.visual_step = 3 
        self.is_animating = False
        random.seed()

    def set_window(self, window):
        self.win = window

    def start_simulation(self):
        print('Simulation started!')  

        for it in range(self.max_iter):
            print('ITERATION: %d' % it)
            
            # Show every 'visual step' iteration
            if it % self.visual_step == 0:
                self.is_animating = True
                # Draw movers
                for mover in self.population:
                    mover.draw(self.win)

            # Simulate one generation
            self.simulate();

            # Undraw if it was animating
            if self.is_animating:
                self.is_animating = False
                for mover in self.population:
                    mover.undraw()

            # Calculate and desc order fitness
            # 2d array: (index, fitness)
            fitness = self.order_by_fitness()

            # Population size
            pop_size = len(self.population)

            # Population elite size
            elite_size = int(pop_size * self.elite)

            # elite size is always even
            if elite_size % 2 == 1:
                elite_size -= 1

            # Save elite indexes
            elite_index = fitness[:elite_size,0]

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
        
        # Number of cuts + 1 --> number of segments
        segment_num = 10
    
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
                # Move it
                mover.update()
                # If mover leaves the field, it dies
                if not mover.isInside(self.field):
                    mover.kill()
                # If mover is inside an obstacle, it dies
                for obstacle in self.obstacles:
                    if mover.isInside(obstacle):
                        mover.kill()
                # Reached goal
                if mover.isInside(self.goal):
                    mover.reachedGoal = True

            if self.is_animating:
                update(60)

def main():
    width = 1000
    height = 500
    win = GraphWin("Draw test", width, height, autoflush = False)
    win.setBackground('red')

    fieldMin = 0.1
    fieldMax = 0.8

    botLeft = Point(width * fieldMin, height * fieldMin)
    topRight = Point(width * fieldMax, height * fieldMax)

    # Draw Field
    field = Rectangle(botLeft, topRight)
    field.setFill('cyan')
    field.setOutline('blue')
    field.draw(win)

    # Draw Goal
    x = width * 0.75
    y = height/2

    radius = 25
    goal = np.array([x, y])
    goalPoint = Point(x, y)
    goalCircle = Circle(goalPoint, radius)
    goalPoint.setFill('cyan')
    goalPoint.setOutline('black')
    goalCircle.draw(win)

    goalTrigger = Rectangle(Point(goal[0] - radius, goal[1] - radius),
                            Point(goal[0] + radius, goal[1] + radius))

    # Draw Obstacles
    obstacleNum = 1
    obstacles = [] 

    obstacles.append(Rectangle(Point(200, 0), Point(250, 300)))
    obstacles.append(Rectangle(Point(350, 0), Point(400, 300)))
    obstacles.append(Rectangle(Point(500, 200), Point(550, 400)))
    obstacles.append(Rectangle(Point(650, 100), Point(680, 200)))
    obstacles.append(Rectangle(Point(650, 230), Point(680, 280)))

    for obstacle in obstacles:
        obstacle.setFill('red')
        obstacle.draw(win)

    animalNum = 50
    animalSize = 5

    startX = width * fieldMin + 10 
    startY = height/2

    lifetime = 500
    numOfGenes = 100
    mutation = 0.02
    speed = 3

    movers = [Mover(startX, startY, speed, animalSize, 
                    lifetime, numOfGenes, goal) 
                for i in range(animalNum)]
    
    genetic = Genetic(movers)
    genetic.set_environment(
            field = field, 
            obstacles = obstacles,
            goal = goalTrigger)
    genetic.set_window(win)
    genetic.start_simulation()

    win.close()

if __name__ == '__main__':
    main()
