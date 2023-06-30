from typing import Dict, AnyStr
from search import BFS, DFS, BellmanFord, Dijkstra
import random


def random_search_over_grid():
    """
    Moving in a 2D grid
    1s obstacles
    0s are valid
    2 is the goal space
    """
    grid = [[0, 0, 1],
            [0, 1, 1],
            [0, 0, 2]]
    current_point = 0, 0
    goal=None
    for i in range(len(grid)):
        for j in range(len(grid)):
            if grid[i][j]== 2:
                goal=i,j
                break
        if goal:
            break

    def get_actions(point):
        """ Get valid actions """
        x, y = point
        move_right = None
        move_left = None
        move_up = None
        move_down = None
        # Border and Obstacle Checking
        if x + 1 < 3 and grid[x + 1][y] != 1:
            move_right = 'R'
        if x - 1 > 0 and grid[x - 1][y] != 1:
            move_left = 'L'
        if y + 1 < 3 and grid[x][y + 1] != 1:
            move_up = 'U'
        if y - 1 >= 0 and grid[x][y - 1] != 1:
            move_down = 'D'
        return [i for i in [move_right, move_left, move_up, move_down] if i]

    def move(point, direction):
        """ Move in a given direction """
        x, y = point
        if direction == 'R':
            return x + 1, y
        elif direction == 'L':
            return x - 1, y
        elif direction == 'U':
            return x, y + 1
        elif direction == 'D':
            return x, y - 1

    # Random Search
    for _ in range(100):
        if current_point == goal:
            print('Goal Found')
            break
        actions = get_actions(current_point)
        current_point = move(current_point, random.choice(actions))

def find_shortest_path_on_weighted_graphs():
    # A weighted graph
    graph = {
        'A': {'B': 6, 'C': 4},
        'B': {'A': 2, 'D': 3, 'E': 7, 'F':1},
        'C': {'A': 4, 'F': 1},
        'D': {'B': 3},
        'E': {'B': 7, 'F': 5},
        'F': {'C': 1, 'E': 5}
    }
    print(graph)
    print(f"BFS solution:{BFS(graph=graph).find_shortest_path(start='A', end='F')}")
    print(f"DFS solution:{BFS(graph=graph).find_shortest_path(start='A', end='F')}")
    print(f"BellmanFord solution:{BellmanFord(graph=graph).find_shortest_path(start='A', end='F')}")
    print(f"Dijkstra solution:{Dijkstra(graph=graph).find_shortest_path(start='A', end='F')}")
    print('Add negative cost...')
    # BFS and DFS cannot operate on graphs with negative weights but Bellman-Ford algorithms does
    graph['A']['C'] = -3
    print(f"BellmanFord solution:{BellmanFord(graph=graph).find_shortest_path(start='A', end='F')}")
    print(f"Dijkstra solution:{Dijkstra(graph=graph).find_shortest_path(start='A', end='F')}")

random_search_over_grid()
find_shortest_path_on_weighted_graphs()