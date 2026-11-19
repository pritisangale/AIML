
from collections import deque

# Graph Representation
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': [],
    'D': [],
    'E': ['F'],
    'F': []
}

# Breadth First Search
def bfs(start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    print("BFS Traversal:")

    while queue:
        node = queue.popleft()
        print(node, end=" ")

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Depth First Search
def dfs(node, visited):
    visited.add(node)
    print(node, end=" ")

    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, visited)

# Starting Node
start = "A"

# Execute BFS
bfs(start)

print("\n")

# Execute DFS
print("DFS Traversal:")
dfs(start, set())

print()


#1. from collections import deque imports deque, which is used to create a queue for BFS.
#2. graph = {...} represents the graph using a dictionary. Each node is connected to its neighbouring nodes.
#3. def bfs(start): defines the Breadth First Search function.
#4. visited = set() stores the nodes that are already visited so that we don't visit them again.
#5. queue = deque([start]) creates a queue and inserts the starting node into it.
#6. visited.add(start) marks the starting node as visited.
#7. while queue: repeats the process until the queue becomes empty.
#8. node = queue.popleft() removes the first node from the queue.
#9. print(node, end=" ") prints the current node.
#10. for neighbor in graph[node]: checks all neighbouring nodes.
#11. if neighbor not in visited: checks whether the neighbour is already visited. If not, it marks it visited and adds it to the queue.
#12. def dfs(node, visited): defines the Depth First Search function.
#13. visited.add(node) marks the current node as visited.
#14. print(node, end=" ") prints the current node.
#15. for neighbor in graph[node]: checks all neighbouring nodes.
#16. if neighbor not in visited: checks whether the neighbour is visited. If not, dfs() calls itself to visit that neighbour. This is called recursion.
#17. start = "A" selects A as the starting node.
#18. bfs(start) calls the BFS function.
#19. dfs(start, set()) calls the DFS function with an empty visited set.
