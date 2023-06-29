
class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]

    def _hash_function(self, key):
        return sum(ord(char) for char in str(key)) % self.size

    def insert(self, key, value):
        index = self._hash_function(key)
        self.table[index].append((key, value))

    def get(self, key):
        index = self._hash_function(key)
        bucket = self.table[index]
        for k, v in bucket:
            if k == key:
                return v
        return None

    def remove(self, key):
        index = self._hash_function(key)
        bucket = self.table[index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                return
        print("Key not found in the hash table.")
