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
linked_list()
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
stacks_and_queues()

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
hash_table()

def binary_tree():
    # Test the code
    bst = BinarySearchTree()
    print(bst)
    [bst.insert(i) for i in [30, 70, 50, 20, 40, 7, 60]]
    print(bst)

    print('Min', bst.min())
    print('Max', bst.max())
    print('Find node', bst.search(40))

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
binary_tree()