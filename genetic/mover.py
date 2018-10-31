import sys
import itertools as itool
from graphics import *
import random
import numpy as np

class Mover():
    def __init__(self, x, y, speed, radius, lifetime, numOfGenes, mutation, goal):
        self.start = np.array([x, y])
        self.circle = Circle(Point(x, y), radius)
        self.circle.setFill('black')
        self.alive = True
        self.goal = np.array([goal.x, goal.y])
        self.life = lifetime
        self.lifetime = lifetime
        self.numOfGenes = numOfGenes
        self.speed = speed
        self.mutation = mutation
        self.eliteCoef = 0.15
        self.dieCoef = 0.2
        self.reachedGoal = False

        self.chromosome = Mover.random_vectors(numOfGenes, -speed, speed)
        self.vel = np.array([0., 0.]) 
        self.acc = self.chromosome[0]
        self.accIndex = 0

    def clone(self):
        return Mover(self.start[0], self.start[1], self.speed,
                self.circle.getRadius(), self.lifetime, self.numOfGenes, 
                self.mutation, Point(self.goal[0], self.goal[1]))

    def update(self):
        self.life -= 1
        if self.life < 0:
            self.kill()

        self.vel += self.acc
        self.accIndex = (self.accIndex + 1) % len(self.chromosome)
        self.acc = self.chromosome[self.accIndex]
        self.move()

    def fitness(self):
        fit = 1.0 / np.linalg.norm(self.goal - self.position())
        if self.reachedGoal:
            fit += 1
        return fit

    def position(self):
        center = self.circle.getCenter()
        return np.array([center.x, center.y])

    def reset(self):
        toStart = self.start - self.position()
        self.circle.move(toStart[0], toStart[1])
        self.life = self.lifetime
        self.alive = True
        self.vel = np.array([0., 0.])
        self.reachedGoal = False

    def draw(self, win):
        self.circle.draw(win)

    def undraw(self):
        self.circle.undraw()

    def mutate(self):
        # random broj
        if random.uniform(0, 1) <= self.mutation:
            return

        size = np.size(self.chromosome, 0)
        num = int(0.01 * size)

        start = random.randint(
                0,
                np.size(self.chromosome, 0) - num - 1)

        for i in range(num):        
            self.chromosome[start + i][0] = \
                    random.uniform(-1, 1)
            self.chromosome[start + i][1] = \
                    random.uniform(-1, 1)

    @staticmethod
    def reproduce(movers):
        coef = movers[0].eliteCoef
        size = len(movers)
        
        elite = Mover.pick_best(movers, int(coef * size))

        newGeneration = []
        for i, fitness in elite:
            newGeneration.append(movers[int(i)])

        for i in range(int((len(movers) - len(elite))/2)):
            parents = Mover.selection(movers)
            child1, child2 = Mover.crossover(parents)

            for child in child1, child2:
                child.reset()
                child.mutate()
                newGeneration.append(child)

        return newGeneration

    @staticmethod
    def crossover(parents):
        child1, child2 = parents[0].clone(), parents[1].clone() 
        cutPoint = int(np.size(child1.chromosome, 0) / 2)
    
        chromLen = np.size(child1.chromosome, 0)
        for i in range(chromLen):
            gene1, gene2 = \
                    parents[0].chromosome[i], parents[1].chromosome[i]
            if i < chromLen /2:
                child1.chromosome[i] = gene1
                child2.chromosome[i] = gene2
            else:
                child1.chromosome[i] = gene2
                child2.chromosome[i] = gene1

        return child1, child2

    @staticmethod
    def pick_best(movers, num):
        fitness = np.array(
                [np.array([i, x.fitness()]) for i, x in enumerate(movers)])

        sumFitness = sum([x.fitness() for x in movers])

        # Normalization of values
        fitness[:,1] /= sumFitness

        # Sorted by normalized fitness values
        normDesc = np.array(
                sorted(fitness, key=lambda x: x[1], reverse=True))

        return normDesc[:num]

    @staticmethod
    def selection(movers):
        fitness = np.array(
                [np.array([i, x.fitness()]) for i, x in enumerate(movers)])

        sumFitness = sum([x.fitness() for x in movers])

        # Normalization of values
        fitness[:,1] /= sumFitness

        # Sorted by normalized fitness values
        normDesc = np.array(
                sorted(fitness, key=lambda x: x[1], reverse=True))

        worstNum = int(movers[0].dieCoef * len(movers))

        normDesc[:-worstNum]

        # Accumulated values
        acc = itool.accumulate(normDesc[:,1])

        for i, newValue in enumerate(acc):
            normDesc[i][1] = newValue 

        select1 = random.uniform(0, 1)
        select2 = random.uniform(0, 1)

        index1 = int(Mover.select_from_pool(normDesc, select1))
        index2 = int(Mover.select_from_pool(normDesc, select2))

        return movers[index1], movers[index2]

    @staticmethod
    def select_from_pool(pool, selection):
        for x in pool:
            if x[1] >= selection:
                return x[0]
        return pool[-1][0]

    @staticmethod
    def random_vectors(length, minValue, maxValue):
        vectors = np.repeat([[0., 0.]], length, axis = 0)
        for i in range(length):
            x = random.uniform(minValue, maxValue)
            y = random.uniform(minValue, maxValue)
            vectors[i][0] = x
            vectors[i][1] = y
                     
        return vectors
        
    def move(self):
        if self.alive:
            self.circle.move(self.vel[0], self.vel[1])

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
    goal = Point(x, y)
    goalCircle = Circle(goal, radius)
    goal.setFill('cyan')
    goal.setOutline('black')
    goalCircle.draw(win)

    goalTrigger = Rectangle(Point(goal.x - radius, goal.y - radius),
                            Point(goal.x + radius, goal.y + radius))

    # Draw Obstacles
    obstacleNum = 1
    obstacles = [] 

    obstacles.append(Rectangle(Point(200, 0), Point(250, 300)))
    #obstacles.append(Rectangle(Point(300, 200), Point(350, 800)))
    obstacles.append(Rectangle(Point(350, 0), Point(400, 300)))
    obstacles.append(Rectangle(Point(500, 200), Point(550, 800)))

    for obstacle in obstacles:
        obstacle.setFill('red')
        obstacle.draw(win)

    random.seed()

    animalNum = 100
    animalSize = 5

    startX = width * fieldMin + 10 
    startY = height/2

    lifetime = 500
    numOfGenes = 100
    mutation = 0.4
    speed = 3

    movers = [Mover(startX, startY, speed, animalSize, 
                    lifetime, numOfGenes, mutation, goal) 
                for i in range(animalNum)]


    fitAvg = 0
    fittest = width
    fitAvgGoal = 1.0/50
    
    generationCount = 1
    end = False
    #while fitAvg < fitAvgGoal:
    while generationCount < 500:
        # Drawing movers
        for mover in movers:
            mover.draw(win)

        anyAlive = True

        while anyAlive:
            # Update
            for mover in movers:
                mover.update()
                if not mover.isInside(field):
                    mover.kill()

                for obstacle in obstacles:
                    if mover.isInside(obstacle):
                        mover.kill()

                if mover.isInside(goalTrigger):
                    print('GOAL')
                    mover.reachedGoal = True


            update(300)

            anyAlive = any(map(lambda x: x.alive, movers))

            if win.checkKey() == 'q':
                print('END')
                end = True
                break

        if end:
            break

        fitness = list(map(lambda x : x.fitness(), movers))
        fitAvg = sum(fitness)/len(fitness)
        fittest = min(fitness)

        print('Generation: ', generationCount)
        print('Average: ', fitAvg)
        print('Fittest: ', fittest)

        newGen = Mover.reproduce(movers)
        for mover in movers:
            mover.undraw()

        movers = newGen

        anyAlive = True
        generationCount += 1


    win.close()

if __name__ == '__main__':
    main()
