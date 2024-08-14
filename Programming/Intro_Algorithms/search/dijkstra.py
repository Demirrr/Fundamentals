from typing import Dict, AnyStr, Tuple, List
import heapq


class Dijkstra:
    def __init__(self, graph: Dict[AnyStr, List]):
        self.graph = graph

    def reconstruct_path(self, start, end, prev):
        path = []
        current = end
        while current != start:
            path.append(current)
            current = prev[current]
        return [start] + list(reversed(path))

    def find_shortest_path(self, start, end):
        # Check if start or end vertex is not in the graph
        if start not in self.graph or end not in self.graph:
            raise KeyError(f'start node:{start} or end node:{end} is not found')
        # (1) Initialization.
        distances = {vertex: float('inf') for vertex in self.graph}
        # Distance to itself is 0
        distances[start] = 0
        prev = {vertex: None for vertex in self.graph}
        # Priority queue to store vertices with their distances
        queue = [(0, start)]
        while queue:
            # Step 5: Extract minimum distance vertex
            dist, u = heapq.heappop(queue)
            if dist > distances[u]:
                continue
            for _ in range(len(self.graph) - 1):
                for vertex in self.graph:
                    # (2.2.) Iterate over neighbours of each vertex.
                    for neighbor, weight in self.graph[vertex].items():
                        if distances[vertex] + weight < distances[neighbor]:
                            distances[neighbor] = distances[vertex] + weight
                            heapq.heappush(queue, (distances[neighbor], neighbor))
                            prev[neighbor] = vertex

        shortest_path = self.reconstruct_path(start, end, prev)
        cost = 0
        for idx, vertex in enumerate(shortest_path):
            if idx == (len(shortest_path) - 1):
                break
            cost += self.graph[vertex][shortest_path[idx + 1]]

        return self.reconstruct_path(start, end, prev), cost
