from graphics import *
import numpy as np
import random
from genetic import *
from mover import *

def create_obstacle(x, y, width, height):
    return Rectangle(Point(x, y), Point(x + width, y + height))

def main():
    width = 1200
    height = 800
    win = GraphWin("Draw test", width, height, autoflush = False)
    win.setBackground('red')

    fieldMin = 0.1
    fieldMax = 0.8

    x_min = width * fieldMin
    x_max = width * fieldMax
    y_min = height * fieldMin
    y_max = height * fieldMax

    botLeft = Point(x_min, y_min)
    topRight = Point(x_max, y_max)

    # Draw Field
    field = Rectangle(botLeft, topRight)
    field.setFill('cyan')
    field.setOutline('blue')
    field.draw(win)

    # Draw Obstacles
    obst_num = 6
    obst_x = 50
    obst_y = 200
    obst_offset = 50

    left = x_min + obst_x + obst_offset
    right = x_max - obst_x - obst_offset
    step = (right-left) // obst_num
    xs = range(int(left), int(right), int(step))

    obstacles = [] 
    for x in xs:
        obstacles.append(
                create_obstacle(
                    x,
                    random.randint(y_min, y_max - obst_y),
                    obst_x,
                    obst_y))

    # Draw obstacles
    for obstacle in obstacles:
        obstacle.setFill('red')
        obstacle.draw(win)

    # Draw Goal
    x = width * 0.75
    y = height/2

    radius = 25
    goal = np.array([x, y])
    goal_point = Point(x, y)
    goal_circle = Circle(goal_point, radius)
    goal_circle.setFill('yellow')
    goal_circle.setOutline('black')
    goal_circle.draw(win)

    goal_rect = Rectangle(Point(goal[0] - radius, goal[1] - radius),
                          Point(goal[0] + radius, goal[1] + radius))
    animal_num = 50
    animal_size = 5

    startX = width * fieldMin + 10 
    startY = height/2
    startRadius = 5

    startRect = Rectangle(Point(startX - radius, startY - radius),
                          Point(startX + radius, startY + radius))
    startRect.setFill('green')
    startRect.setOutline('black')
    startRect.draw(win)

    lifetime = 100
    gene_num = 100
    speed = 3

    movers = [Mover(startX, startY, speed, animal_size, 
                    lifetime, gene_num, goal) 
                for i in range(animal_num)]
    
    genetic = Genetic(movers)
    genetic.set_environment(
            field = field, 
            obstacles = obstacles,
            goal = goal_rect)

    genetic.set_window(win)
    genetic.start_simulation()

    win.close()

if __name__ == '__main__':
    main()
