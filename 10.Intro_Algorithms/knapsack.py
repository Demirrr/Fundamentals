def knapsack(weights, values, capacity):
    n = len(weights)
    # Create a table to store the maximum values for each subproblem
    table = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            # If the current item's weight is less than or equal to the current capacity
            if weights[i - 1] <= w:
                # Choose the maximum between including and excluding the current item
                table[i][w] = max(values[i - 1] + table[i - 1][w - weights[i - 1]], table[i - 1][w])
            else:
                # If the current item's weight is greater than the current capacity, exclude it
                table[i][w] = table[i - 1][w]

    # Find the items included in the knapsack
    included_items = []
    i, w = n, capacity
    while i > 0 and w > 0:
        if table[i][w] != table[i - 1][w]:
            included_items.append(i - 1)
            w -= weights[i - 1]
        i -= 1

    # Return the maximum value and the included items
    return table[n][capacity], included_items


# Example usage
weights = [3, 2, 4, 1]
values = [5, 3, 8, 2]
capacity = 6

max_value, items = knapsack(weights, values, capacity)
print("Maximum value:", max_value)
print("Included items:", items)


def knapsack_greedy(weights, values, capacity):
    # Calculate the value-to-weight ratio for each item
    value_per_weight = [(values[i] / weights[i], i) for i in range(len(values))]

    # Sort items in descending order based on value-to-weight ratio
    value_per_weight.sort(reverse=True)

    total_value = 0
    selected_items = []

    for vw_ratio, item_index in value_per_weight:
        item_weight = weights[item_index]
        item_value = values[item_index]

        if capacity >= item_weight:
            # Take the whole item
            total_value += item_value
            selected_items.append(item_index)
            capacity -= item_weight

    return total_value, selected_items



max_value, selected_items = knapsack_greedy(weights, values, capacity)

print(f"Maximum value: {max_value}")
print("Selected items:", selected_items)
