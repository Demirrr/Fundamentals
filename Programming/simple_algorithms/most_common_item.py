# Input: list of items
# Output: Most common item
import sys
from collections import Counter
# Inputs

x=[1,1,1,4,3]
counter=Counter(x)
item,freq=counter.most_common()[0]
print(f'Input:{x}\tMost common item:{item}\t Frequency:{freq}')


x=['a','b','b','c']
counter=Counter(x)
item,freq=counter.most_common()[0]
print(f'Input:{x}\tMost common item:{item}\t Frequency:{freq}')

