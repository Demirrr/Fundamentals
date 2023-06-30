from typing import Dict,AnyStr
class Stack:
    """ Last-in, First-out"""
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        else:
            print("Stack is empty.")

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        else:
            print("Stack is empty.")

    def size(self):
        return len(self.items)

class DFS:
    def __init__(self, graph: Dict[AnyStr, Dict[AnyStr, int]],depth=10):
        self.graph = graph
        # @TODO
        self.depth=depth

    def find_shortest_path(self, start, end):
        # Check if start or end vertex is not in the graph
        if start not in self.graph or end not in self.graph:
            raise KeyError(f'start node:{start} or end node:{end} is not found')
        # Track visited vertices
        visited = set()
        # Initialize the queue with the start vertex and path
        stack= Stack()
        stack.push((start, [start]))
        sortest_path = None
        while stack:
            vertex, path = stack.pop()
            if vertex == end:
                sortest_path = path
                break

            if vertex not in visited:
                visited.add(vertex)

                # Explore adjacent vertices
                for neighbor, weight in self.graph[vertex].items():
                    if neighbor not in visited:
                        stack.push((neighbor, path + [neighbor]))

        total_dist=0
        for ind, v in enumerate(sortest_path):
            if ind>=1:
                total_dist+=self.graph[sortest_path[ind-1]][v]
        return sortest_path, total_dist
