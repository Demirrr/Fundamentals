from typing import Dict, AnyStr


class Queue:
    """ First-in, First-out"""

    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        else:
            print("Queue is empty.")

    def peek(self):
        if not self.is_empty():
            return self.items[0]
        else:
            print("Queue is empty.")

    def size(self):
        return len(self.items)


class BFS:
    def __init__(self, graph: Dict[AnyStr, Dict[AnyStr, int]]):
        self.graph = graph

    def find_shortest_path(self, start, end):
        # (1) Check if start or end vertex is not in the graph
        if start not in self.graph or end not in self.graph:
            raise KeyError(f'start node:{start} or end node:{end} is not found')
        # (2) Track visited vertices
        visited = set()
        # (3) Initialize the queue with the start vertex and path
        queue = Queue()
        queue.enqueue((start, [start]))
        # (4)
        sortest_path = None
        while queue:
            vertex, path = queue.dequeue()
            # Terminate if it is the destination.
            if vertex == end:
                sortest_path = path
                break
            # Add into seen.
            if vertex not in visited:
                visited.add(vertex)
                # Explore adjacent vertices
                for neighbor, weight in self.graph[vertex].items():
                    if neighbor not in visited:
                        queue.enqueue((neighbor, path + [neighbor]))
        total_dist=0
        for ind, v in enumerate(sortest_path):
            if ind>=1:
                total_dist+=self.graph[sortest_path[ind-1]][v]
        return sortest_path, total_dist
