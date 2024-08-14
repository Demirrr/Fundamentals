"""binary_tree.py"""


class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def depth(self):
        left, right = self.left, self.right
        depth = 1
        while left is not None:
            depth += 1
            left = left.left
        return depth

    def __str__(self):
        value = "\t" * self.depth() + f"({self.val})"
        if self.left is not None:
            value+=f"\n{self.left} {self.right.val}"

        return value


def invertTree(root):
    """
    post order traversal over a binary tree
    Go left subtree until its exhausted

    Time complexity : O(N)
    Space complexity: O(N)

    """
    if root is None:
        return None
    invertTree(root.left)
    invertTree(root.right)
    root.left, root.right = root.right, root.left
    return root


binary_tree = TreeNode(4,
                       left=TreeNode(2,
                                     left=TreeNode(1),
                                     right=TreeNode(3)),
                       right=TreeNode(7,
                                      left=TreeNode(3),
                                      right=TreeNode(9)))
print("Binary tree")
print(binary_tree)
print("Inverted Binary tree")
print(invertTree(binary_tree))
