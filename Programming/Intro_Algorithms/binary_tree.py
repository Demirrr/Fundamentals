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

AVL Search Tree:

"""

import random


class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None
        # used in AVL
        self.height = 1

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

    def insert(self, key: int):
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

    def min(self, current=None, iterative=False):
        """ Finds and returns the minimum node"""
        if current is None:
            current = self.root

        if iterative:
            while current.left:
                current = current.left
            return current.key

        if current.left is None:
            return current.key
        else:
            return self.min(current.left)

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
        """ Bad news it is a list :)

                         ()
                        /
                      ()
                     /
                    ()
                    /
                 ()

        """
        results = []
        current = self._search_recursive(current=self.root, key=k)
        results.append(current)
        while current is not None and current.left:
            current = current.left
            results.append(current)
        return results


class BSTNode:
    """A node in the vanilla BST tree."""

    def __init__(self, parent, k):
        """Creates a node.

        Args:
            parent: The node's parent.
            k: key of the node.
        """
        self.key = k
        self.parent = parent
        self.left = None
        self.right = None

    def _str(self):
        """Internal method for ASCII art."""
        label = str(self.key)
        if self.left is None:
            left_lines, left_pos, left_width = [], 0, 0
        else:
            left_lines, left_pos, left_width = self.left._str()
        if self.right is None:
            right_lines, right_pos, right_width = [], 0, 0
        else:
            right_lines, right_pos, right_width = self.right._str()
        middle = max(right_pos + left_width - left_pos + 1, len(label), 2)
        pos = left_pos + middle // 2
        width = left_pos + middle + right_width - right_pos
        while len(left_lines) < len(right_lines):
            left_lines.append(' ' * left_width)
        while len(right_lines) < len(left_lines):
            right_lines.append(' ' * right_width)
        if (middle - len(label)) % 2 == 1 and self.parent is not None and \
                self is self.parent.left and len(label) < middle:
            label += '.'
        label = label.center(middle, '.')
        if label[0] == '.': label = ' ' + label[1:]
        if label[-1] == '.': label = label[:-1] + ' '
        lines = [' ' * left_pos + label + ' ' * (right_width - right_pos),
                 ' ' * left_pos + '/' + ' ' * (middle - 2) +
                 '\\' + ' ' * (right_width - right_pos)] + \
                [left_line + ' ' * (width - left_width - right_width) + right_line
                 for left_line, right_line in zip(left_lines, right_lines)]
        return lines, pos, width

    def __str__(self):
        return '\n'.join(self._str()[0])

    def find(self, k):
        """Finds and returns the node with key k from the subtree rooted at this
        node.

        Args:
            k: The key of the node we want to find.

        Returns:
            The node with key k.
        """
        if k == self.key:
            return self
        elif k < self.key:
            if self.left is None:
                return None
            else:
                return self.left.find(k)
        else:
            if self.right is None:
                return None
            else:
                return self.right.find(k)

    def min(self):
        """Finds the node with the minimum key in the subtree rooted at this
        node.

        Returns:
            The node with the minimum key.
        """
        current = self
        while current.left is not None:
            current = current.left
        return current
    def max(self):
        current = self
        while current.right is not None:
            current = current.right
        return current

    def next_larger(self):
        """Returns the node with the next larger key (the successor) in the BST.
        """
        if self.right is not None:
            return self.right.min()
        current = self
        while current.parent is not None and current is current.parent.right:
            current = current.parent
        return current.parent

    def insert(self, node):
        """Inserts a node into the subtree rooted at this node.

        Args:
            node: The node to be inserted.
        """
        if node is None:
            return
        if node.key < self.key:
            if self.left is None:
                node.parent = self
                self.left = node
            else:
                self.left.insert(node)
        else:
            if self.right is None:
                node.parent = self
                self.right = node
            else:
                self.right.insert(node)

    def delete(self):
        """Deletes and returns this node from the BST."""
        if self.left is None or self.right is None:
            if self is self.parent.left:
                self.parent.left = self.left or self.right
                if self.parent.left is not None:
                    self.parent.left.parent = self.parent
            else:
                self.parent.right = self.left or self.right
                if self.parent.right is not None:
                    self.parent.right.parent = self.parent
            return self
        else:
            s = self.next_larger()
            self.key, s.key = s.key, self.key
            return s.delete()

    def check_ri(self):
        """Checks the BST representation invariant around this node.

        Raises an exception if the RI is violated.
        """
        if self.left is not None:
            if self.left.key > self.key:
                raise RuntimeError("BST RI violated by a left node key")
            if self.left.parent is not self:
                raise RuntimeError("BST RI violated by a left node parent "
                                   "pointer")
            self.left.check_ri()
        if self.right is not None:
            if self.right.key < self.key:
                raise RuntimeError("BST RI violated by a right node key")
            if self.right.parent is not self:
                raise RuntimeError("BST RI violated by a right node parent "
                                   "pointer")
            self.right.check_ri()


class BST:
    """A binary search tree."""

    def __init__(self, klass=BSTNode):
        """Creates an empty BST.

        Args:
            klass (optional): The class of the node in the BST. Default to
                BSTNode.
        """
        self.root = None
        self.klass = klass

    def __str__(self):
        if self.root is None: return '<empty tree>'
        return str(self.root)

    def find(self, k):
        """Finds and returns the node with key k from the subtree rooted at this
        node.

        Args:
            k: The key of the node we want to find.

        Returns:
            The node with key k or None if the tree is empty.
        """
        return self.root and self.root.find(k)

    def search(self, k):
        """Finds and returns the node with key k from the subtree rooted at this
        node.

        Args:
            k: The key of the node we want to find.

        Returns:
            The node with key k or None if the tree is empty.
        """
        return self.root and self.root.find(k)

    def min(self):
        """Returns the minimum node of this BST."""
        return self.root and self.root.min()

    def max(self):
        """Returns the max node of this BST."""
        return self.root and self.root.max()

    def insert(self, k):
        """Inserts a node with key k into the subtree rooted at this node.

        Args:
            k: The key of the node to be inserted.

        Returns:
            The node inserted.
        """
        node = self.klass(None, k)
        if self.root is None:
            # The root's parent is None.
            self.root = node
        else:
            self.root.insert(node)
        return node

    def delete(self, k):
        """Deletes and returns a node with key k if it exists from the BST.

        Args:
            k: The key of the node that we want to delete.

        Returns:
            The deleted node with key k.
        """
        node = self.find(k)
        if node is None:
            return None
        if node is self.root:
            pseudoroot = self.klass(None, 0)
            pseudoroot.left = self.root
            self.root.parent = pseudoroot
            deleted = self.root.delete()
            self.root = pseudoroot.left
            if self.root is not None:
                self.root.parent = None
            return deleted
        else:
            return node.delete()

    def next_larger(self, k):
        """Returns the node that contains the next larger (the successor) key in
        the BST in relation to the node with key k.

        Args:
            k: The key of the node of which the successor is to be found.

        Returns:
            The successor node.
        """
        node = self.find(k)
        return node and node.next_larger()

    def check_ri(self):
        """Checks the BST representation invariant.

        Raises:
            An exception if the RI is violated.
        """
        if self.root is not None:
            if self.root.parent is not None:
                raise RuntimeError("BST RI violated by the root node's parent "
                                   "pointer.")
            self.root.check_ri()


def height(node):
    if node is None:
        return -1
    else:
        return node.height


def update_height(node):
    node.height = max(height(node.left), height(node.right)) + 1


class AVL(BST):
    """ AVL binary search tree implementation.
    Supports insert, delete, find, find_min, next_larger each in O(lg n) time."""

    def left_rotate(self, x):
        y = x.right
        y.parent = x.parent
        if y.parent is None:
            self.root = y
        else:
            if y.parent.left is x:
                y.parent.left = y
            elif y.parent.right is x:
                y.parent.right = y
        x.right = y.left
        if x.right is not None:
            x.right.parent = x
        y.left = x
        x.parent = y
        update_height(x)
        update_height(y)

    def right_rotate(self, x):
        y = x.left
        y.parent = x.parent
        if y.parent is None:
            self.root = y
        else:
            if y.parent.left is x:
                y.parent.left = y
            elif y.parent.right is x:
                y.parent.right = y
        x.left = y.right
        if x.left is not None:
            x.left.parent = x
        y.right = x
        x.parent = y
        update_height(x)
        update_height(y)

    def rebalance(self, node):
        while node is not None:
            update_height(node)
            if height(node.left) >= 2 + height(node.right):
                if height(node.left.left) >= height(node.left.right):
                    self.right_rotate(node)
                else:
                    self.left_rotate(node.left)
                    self.right_rotate(node)
            elif height(node.right) >= 2 + height(node.left):
                if height(node.right.right) >= height(node.right.left):
                    self.left_rotate(node)
                else:
                    self.right_rotate(node.right)
                    self.left_rotate(node)
            node = node.parent

    ## find(k), find_min(), and next_larger(k) inherited from bst.BST

    def insert(self, k):
        """Inserts a node with key k into the subtree rooted at this node.
        This AVL version guarantees the balance property: h = O(lg n).

        Args:
            k: The key of the node to be inserted.
        """
        node = super(AVL, self).insert(k)
        self.rebalance(node)

    def delete(self, k):
        """Deletes and returns a node with key k if it exists from the BST.
        This AVL version guarantees the balance property: h = O(lg n).

        Args:
            k: The key of the node that we want to delete.

        Returns:
            The deleted node with key k.
        """
        node = super(AVL, self).delete(k)
        ## node.parent is actually the old parent of the node,
        ## which is the first potentially out-of-balance node.
        self.rebalance(node.parent)


def search_tree():
    bst = BinarySearchTree()
    bst2 = BST()
    avl = AVL()
    # (1)
    items = [random.randrange(100) for i in range(10)]
    print(f"{bst}\n{bst2}\n{avl}")
    # (2) Insert.
    for item in items:
        bst.insert(item)
        bst2.insert(item)
        avl.insert(item)
    print(f"{bst}\n\n{bst2}\n\n{avl}")

    for t in [bst, bst2, avl]:
        # (3) Find the minimum and maximum.
        print('Min', t.min())
        print('Max', t.max())
        # (4) Find.
        print('Find node', t.search(40))
        # (5) Delete.
        print("Delete 30:")
        t.delete(30)


search_tree()
