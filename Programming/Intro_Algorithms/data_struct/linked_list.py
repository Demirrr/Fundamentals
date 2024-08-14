class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head is None

    def append(self, data):
        new_node = Node(data)

        if self.is_empty():
            self.head = new_node
        else:
            current = self.head
            # iterate until there is no next
            while current.next:
                current = current.next
            current.next = new_node

    def prepend(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def insert_after(self, key, data):
        new_node = Node(data)
        current = self.head

        while current:
            if current.data == key:
                new_node.next = current.next
                current.next = new_node
                return
            current = current.next

        print(f"Key '{key}' not found in the linked list.")

    def delete(self, key):
        if self.is_empty():
            print("Linked list is empty.")
            return

        if self.head.data == key:
            self.head = self.head.next
            return

        current = self.head
        prev = None

        while current:
            if current.data == key:
                prev.next = current.next
                return
            prev = current
            current = current.next

        print(f"Key '{key}' not found in the linked list.")

    def display(self):
        if self.is_empty():
            print("Linked list is empty.")
            return

        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")
