from typing import Dict, AnyStr


class BellmanFord:
    """
    The Bellman-Ford algorithm solves the single-source shortest-paths problem in
    the general case in which edge weights may be negative.



    """

    def __init__(self, graph: Dict[AnyStr, Dict[AnyStr, int]]):
        self.graph = graph

    def compute_distances(self, start, end=None):
        # (1) Initialization.
        distances = {vertex: float('inf') for vertex in self.graph}
        # Distance to itself is 0
        distances[start] = 0
        prev = {vertex: None for vertex in self.graph}
        # (2) Relaxation.
        # (2.1) Iterate over N-1 times.
        for _ in range(len(self.graph) - 1):
            # (2.1) Iterate over each vertex. O(N)
            for vertex in self.graph:
                # (2.2.) Iterate over neighbours of each vertex.
                for neighbor, weight in self.graph[vertex].items():
                    if distances[vertex] + weight < distances[neighbor]:
                        distances[neighbor] = distances[vertex] + weight
                        prev[neighbor] = vertex
        # (3) Check for negative cycles. O(N)
        for vertex in self.graph:
            for neighbor, weight in self.graph[vertex].items():
                if distances[vertex] + weight < distances[neighbor]:
                    # Negative cycle found
                    raise RuntimeError("Negative cycle found in the graph")
        return distances, self.reconstruct_path(start, end, prev)

    def reconstruct_path(self, start, end, prev):
        path = []
        current = end
        while current != start:
            path.append(current)
            current = prev[current]
        return [start] + list(reversed(path))

    def find_distance(self, start):
        # Check if start or end vertex is not in the graph
        if start not in self.graph:
            raise KeyError(f'start node:{start} or end node:{end} is not found')
        return self.compute_distances(start)

    def find_shortest_path(self, start, end):
        # Check if start or end vertex is not in the graph
        if start not in self.graph or end not in self.graph:
            raise KeyError(f'start node:{start} or end node:{end} is not found')
        # (1) Initialization.
        distances = {vertex: float('inf') for vertex in self.graph}
        # Distance to itself is 0
        distances[start] = 0
        prev = {vertex: None for vertex in self.graph}
        # (2) Relaxation.
        # (2.1) Iterate over N-1 times.
        for _ in range(len(self.graph) - 1):
            # (2.1) Iterate over each vertex. O(N)
            for vertex in self.graph:
                # (2.2.) Iterate over neighbours of each vertex.
                for neighbor, weight in self.graph[vertex].items():
                    if distances[vertex] + weight < distances[neighbor]:
                        distances[neighbor] = distances[vertex] + weight
                        prev[neighbor] = vertex
        # (3) Check for negative cycles. O(N)
        for vertex in self.graph:
            for neighbor, weight in self.graph[vertex].items():
                if distances[vertex] + weight < distances[neighbor]:
                    # Negative cycle found
                    raise RuntimeError("Negative cycle found in the graph")

        shortest_path=self.reconstruct_path(start, end, prev)
        cost=0
        for idx, vertex in enumerate(shortest_path):
            if idx==(len(shortest_path)-1):
                break
            cost+=self.graph[vertex][shortest_path[idx+1]]

        return self.reconstruct_path(start, end, prev),cost



