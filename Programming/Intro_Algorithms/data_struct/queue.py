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
