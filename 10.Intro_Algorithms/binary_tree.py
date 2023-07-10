""" Example for the motivation

• Runway reservation system

Schedule a landing upon a landing time request (t) provided that
1. t is later than the very next scheduled landing time request (nst).
2. there is no scheduled landing time request (st) within a k minutes

A naive solution

1. Check t is greater than R[0]
2. Iterate over R to check this abs(t,R[i]) > k holds
3. Append t into R
4. Sort R.


Sorted List :  Appending and sorting takes O(n log (n))
Dictionary: Insertion O(1) and k minute check O(n)

Binary Search Tree:
All operations take O(h) where h denotes the height of the BST.
"""


class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None

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

    def __str__(self):
        if self.root is None: return '<empty tree>'

        def recurse(node):
            if node is None: return [], 0, 0
            label = str(node.key)
            left_lines, left_pos, left_width = recurse(node.left)
            right_lines, right_pos, right_width = recurse(node.right)
            middle = max(right_pos + left_width - left_pos + 1, len(label), 2)
            pos = left_pos + middle // 2
            width = left_pos + middle + right_width - right_pos
            while len(left_lines) < len(right_lines):
                left_lines.append(' ' * left_width)
            while len(right_lines) < len(left_lines):
                right_lines.append(' ' * right_width)
            if (middle - len(label)) % 2 == 1 and node.parent is not None and \
                    node is node.parent.left and len(label) < middle:
                label += '.'
            label = label.center(middle, '.')
            if label[0] == '.': label = ' ' + label[1:]
            if label[-1] == '.': label = label[:-1] + ' '
            lines = [' ' * left_pos + label + ' ' * (right_width - right_pos),
                     ' ' * left_pos + '/' + ' ' * (middle - 2) +
                     '\\' + ' ' * (right_width - right_pos)] + \
                    [left_line + ' ' * (width - left_width - right_width) +
                     right_line
                     for left_line, right_line in zip(left_lines, right_lines)]
            return lines, pos, width

        return '\n'.join(recurse(self.root)[0])

    def insert(self, key=int):
        """ Insert a node into search tree with a given value """

        def insert_recursive(current, k):
            """ Insert a node into search tree with a given value """
            # (1)
            if k < current.key:
                if current.left is None:
                    n = Node(k)
                    current.left = n
                    n.parent = current
                else:
                    insert_recursive(current.left, k)
            # (2)
            elif k > current.key:
                if current.right is None:
                    n = Node(k)
                    current.right = n
                    n.parent = current
                else:
                    insert_recursive(current.right, k)
            else:
                raise RuntimeError('Logic incorrect.')

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

    def less_or_equal(self, k):
        """ O(h) h => heights"""
        results = []
        current = self._search_recursive(current=self.root, key=k)
        results.append(current)
        while current.left:
            current = current.left
            results.append(current)
        return results


# (1) Initialize.
bst = BinarySearchTree()
print(bst)
# (2) Insert.
[bst.insert(i) for i in [30, 70, 50, 20, 40, 7, 60]]
print(bst)
# (3) Find the minimum and maximum.
print('Min', bst.min())
print('Max', bst.max())
# (4) Find.
print('Find node', bst.search(40))
print("Inorder traversal:")
bst.inorder_traversal()
# (5) Delete.
print("Delete 30:")
bst.delete(30)
# (6) (Rank(t): How many planes are scheduled to land at times ≤ t?
print(bst)
for i in bst.less_or_equal(40):
    print(i)
