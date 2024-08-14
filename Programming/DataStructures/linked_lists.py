"""linked_lists.py: Example of merging two sorted singly-linked list


[1,2,4]
[1,2,3]
<__main__.ListNode object at 0x7f218bf60d90> 1 <__main__.ListNode object at 0x7f218bf60ee0>
<__main__.ListNode object at 0x7f218bf60ee0> 1 <__main__.ListNode object at 0x7f218bf60f70>
<__main__.ListNode object at 0x7f218bf60f70> 2 <__main__.ListNode object at 0x7f218bf60df0>
<__main__.ListNode object at 0x7f218bf60df0> 3 <__main__.ListNode object at 0x7f218bf60e50>
<__main__.ListNode object at 0x7f218bf60e50> 4 <__main__.ListNode object at 0x7f218bf60fd0>
<__main__.ListNode object at 0x7f218bf60fd0> 4 None

"""
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, value=0, next_item=None):
        self.value = value
        self.next_item = next_item

def merge(list1, list2):
    current = initial = ListNode()
    # while lists are not exhausted
    while list1 and list2:

        if list1.value < list2.value:
            # Update the current list with list1.
            current.next_item = list1
            list1, current = list1.next_item, list1
        else:
            # Update the current list with list2.
            current.next_item = list2
            list2, current = list2.next_item, list2

    # Detect which
    if list1 is None:
        current.next_item = list2
    else:
        current.next_item = list1
    return initial.next_item

d = merge(ListNode(value=1, next_item=ListNode(value=2, next_item=ListNode(value=4))),
                  ListNode(value=1, next_item=ListNode(value=3, next_item=ListNode(value=4))))
while d:
    print(d, d.value, d.next_item)
    d = d.next_item

