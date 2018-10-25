import sys
import itertools as itool
from graphics import *
import random
import numpy as np

class Mover():
    def __init__(self, x, y, radius, lifetime, goal):
        self.start = np.array([x, y])
        self.circle = Circle(Point(x, y), radius)
        self.circle.setFill('black')
        self.alive = True
        self.goal = np.array([goal.x, goal.y])
        self.life = lifetime
        self.lifetime = lifetime

        self.chromosome = Mover.random_vectors(10, -1, 1)
        self.vel = np.array([0., 0.])
        self.acc = self.chromosome[0]
        self.accIndex = 0

    def clone(self):
        return Mover(self.start[0], self.start[1], 
                self.circle.getRadius(),
                self.lifetime, Point(self.goal[0], self.goal[1]))

    def update(self):
        self.life -= 1
        if self.life < 0:
            self.kill()

        self.vel += self.acc
        self.accIndex = (self.accIndex + 1) % len(self.chromosome)
        self.acc = self.chromosome[self.accIndex]
        self.move()

    def fitness(self):
        return 1.0 / np.linalg.norm(self.goal - self.position())

    def position(self):
        center = self.circle.getCenter()
        return np.array([center.x, center.y])

    def reset(self):
        toStart = self.start - self.position()
        self.circle.move(toStart[0], toStart[1])
        self.life = self.lifetime
        self.alive = True
        self.vel = np.array([0., 0.])

    def draw(self, win):
        self.circle.draw(win)

    def undraw(self):
        self.circle.undraw()

    def mutate(self, chance):
        if random.uniform(0, 1) > chance:
            return

        randomGene = random.randint(
                0,
                np.size(self.chromosome, 0)-1)
        self.chromosome[randomGene][0] = \
                random.uniform(-1, 1)
        self.chromosome[randomGene][1] = \
                random.uniform(-1, 1)

    @staticmethod
    def reproduce(movers):
        newGeneration = []
        for i in range(len(movers)):
            parents = Mover.selection(movers)
            child1, child2 = Mover.crossover(parents)

            for child in child1, child2:
                child.reset()
                child.mutate(0.02)
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
    def selection(movers):
        fitness = np.array(
                [np.array([i, x.fitness()]) 
                    for i, x in enumerate(movers)])

        sumFitness = sum([x.fitness() for x in movers])

        avgFitness = sumFitness / len(movers)
        #normalized = [[i, fit/sumFitness] for i, fit in fitness]

        fitness[:,1] /= sumFitness

        normDesc = np.array(
                sorted(fitness, key=lambda x: x[1], reverse=True))

        print(normDesc[:,1])
        # Accumulated values
        acc = itool.accumulate(normDesc[1,:])
        for x in acc:
            print(x)

        indexPool = []
        for i, fit in fitness:
            indexPool.extend([i] * int(avgFitness/fit))

        first = random.choice(indexPool)
        second = random.choice(indexPool)

        while first == second:
            second = random.choice(indexPool)

        return movers[first], movers[second]

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
    goal = Point(width * 0.75, height/2)
    goalCircle = Circle(goal, 25)
    goal.setFill('cyan')
    goal.setOutline('black')
    goalCircle.draw(win)

    random.seed()

    animalNum = 10
    animalSize = 5

    startX = width * fieldMin + 10 
    startY = height/2

    lifetime = 100

    movers = [Mover(startX, startY, animalSize, lifetime, goal) 
                for i in range(animalNum)]


    fitAvg = width
    fittest = width
    fitAvgGoal = 50
    
    generationCount = 1
    end = False
    while fitAvg > fitAvgGoal:
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

            update(30)
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
