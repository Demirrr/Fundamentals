class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

    def __str__(self):
        return f"Node(key:{self.key})"


class BinarySearchTree:
    """
    The search tree data structure supports many dynamic-set operations, e.g.
    including search, minimum, maximum, predecessor, successor, insert, and delete.
    Thus, we can use a search tree both as a dictionary and as a priority queue.

    For a complete binary tree with n nodes, such operations run in O(log(n))
    in worst-case time.


    A binary search tree can be represented by a linked data structure in which each node is an object.
    """

    def __init__(self):
        self.root: Node = None

    def insert(self, key=int):
        def insert_recursive(current, key):
            if key < current.key:
                if current.left is None:
                    current.left = Node(key)
                else:
                    insert_recursive(current.left, key)
            elif key > current.key:
                if current.right is None:
                    current.right = Node(key)
                else:
                    insert_recursive(current.right, key)

        if self.root is None:
            self.root = Node(key)
        else:
            insert_recursive(self.root, key)

    def delete(self, key):
        self.root = self._delete_recursive(self.root, key)

    def _delete_recursive(self, current, key):
        if current is None:
            return current
        if key < current.key:
            current.left = self._delete_recursive(current.left, key)
        elif key > current.key:
            current.right = self._delete_recursive(current.right, key)
        else:
            if current.left is None:
                return current.right
            elif current.right is None:
                return current.left
            else:
                successor = self._find_min(current.right)
                current.key = successor.key
                current.right = self._delete_recursive(current.right, successor.key)
        return current

    def search(self, key: int, recursive=True):
        if recursive:
            return self._search_recursive(self.root, key)
        else:
            return self._search_iterative(self.root, key)

    def _search_recursive(self, current, key):
        """ O(h) s.t. h is the height. """
        if current is None or current.key == key:
            return current
        if key < current.key:
            return self._search_recursive(current.left, key)
        else:
            return self._search_recursive(current.right, key)

    @staticmethod
    def _search_iterative(current, key):
        while current is not None and current.key != key:
            if key < current.key:
                current = current.left
            else:
                current = current.right
        return current

    @staticmethod
    def _find_min(current):
        while current.left:
            current = current.left
        return current

    def min(self, current=None):
        if current is None:
            current = self.root
        while current.left:
            current = current.left
        return current.key

    def max(self, current=None):
        if current is None:
            current = self.root
        while current.right:
            current = current.right
        return current.key

    def inorder_traversal(self):
        self._inorder_recursive(self.root)

    def _inorder_recursive(self, current):
        if current is not None:
            self._inorder_recursive(current.left)
            print(current.key, end=" ")
            self._inorder_recursive(current.right)


class RBSearchTree:
    """ Red-Black Tree"""
    def __init__(self):
        pass
# Test the code
bst = BinarySearchTree()
[bst.insert(i) for i in [30, 70, 50, 20, 40, 7, 60]]

print(bst.min())
print(bst.max())
print(bst.search(40))
exit(1)

print("Inorder traversal:")
bst.inorder_traversal()
print()

print("Search 40:")
if bst.search(40):
    print("Key found.")
else:
    print("Key not found.")

print("Delete 30:")
bst.delete(30)

print("Inorder traversal after deletion:")
bst.inorder_traversal()
print()
exit(1)

from data_struct import *


def linked_list():
    # Test the code
    linked_list = LinkedList()
    linked_list.append(1)
    linked_list.append(2)
    linked_list.append(3)
    linked_list.prepend(0)
    linked_list.insert_after(2, 2.5)
    linked_list.display()

    linked_list.delete(2)
    linked_list.display()


def stacks_and_queues():
    stack = Stack()
    queue = Queue()
    stack.push(2)
    stack.push(4)
    stack.pop()
    print(stack.items)

    queue.enqueue(2)
    queue.enqueue(4)
    queue.dequeue()
    print(queue.items)


def hash_table():
    # Test the code
    hash_table = HashTable(10)

    # Insert key-value pairs
    hash_table.insert(key="apple", value=5)
    hash_table.insert("banana", 10)
    hash_table.insert("orange", 7)

    # Get values by keys
    print(hash_table.get("apple"))  # Output: 5
    print(hash_table.get("banana"))  # Output: 10
    print(hash_table.get("orange"))  # Output: 7

    # Remove a key-value pair
    hash_table.remove("banana")
    print(hash_table.get("banana"))  # Output: None


linked_list()
stacks_and_queues()
hash_table()
