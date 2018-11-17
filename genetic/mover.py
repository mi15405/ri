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
        self.lifetime = lifetime
        self.speed = speed
        self.circle = Circle(Point(x, y), radius)
        self.circle.setFill('black')
        self.is_drawn = False
        self.goal = goal
        self.numOfGenes = numOfGenes
        self.chromosome = Mover.random_vectors(numOfGenes, -speed, speed)
        self.reset_vel()
        self.acc = self.chromosome[0]
        self.accIndex = 0
        self.reachedGoal = False

    def reset_vel(self):
        self.vel = np.array([0., 0.]) 

    def clone(self):
        return Mover(
                self.start[0], self.start[1], self.speed,
                self.circle.getRadius(), self.lifetime, 
                self.numOfGenes, self.goal)

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
