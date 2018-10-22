from graphics import *
import random

class Animal():
    def __init__(self, x, y, radius):
        self.point = Point(x, y)
        self.circle = Circle(self.point, radius)
        self.circle.setFill('black')

    def draw(self, win):
        self.circle.draw(win)

    def move(self):
        self.circle.move(1, 1)


def randomInts(num, left, right):
    numbers = []
    for i in range(num):
        numbers.append(random.randint(left, right))

    return numbers

def main():
    width = 1200
    height = 800
    win = GraphWin("Draw test", width, height, autoflush = False)

    random.seed()

    animalNum = 100
    animalSize = 10


    randomPos = zip(randomInts(animalNum, 0, width), randomInts(animalNum, 0 , height))

    animals = [Animal(x, y, animalSize) for x, y in randomPos]

    for animal in animals:
        animal.draw(win)

    while True:
        for animal in animals:
            animal.move()

        update(30)

    win.getMouse()
    win.close()

if __name__ == '__main__':
    main()
