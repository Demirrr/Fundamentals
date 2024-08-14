""" check_palindrome.py: Check if a string is a palindrome """
import re


def isPalindrome(s: str):
    s = re.sub(r'[^0-9a-zA-Z]', '', s).lower()
    left_pointer = 0
    right_pointer = len(s) - 1
    while left_pointer < right_pointer:
        if s[left_pointer] != s[right_pointer]:
            return False
        left_pointer += 1
        right_pointer -= 1
    return True


assert isPalindrome(" ") == True
assert isPalindrome("Ada ") == True
assert isPalindrome("A man, a plan,") == False
