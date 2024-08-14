from typing import List

def recursive_activity_selector(start: List[int], finish: List[int], k: int, n: int):
    m = k + 1
    # (1) If the finish time of the last taken activity is greater than the start time of the m.th activity
    while n> m and finish[k] > start[m]:
        m += 1
    # (2) Include the current activity in the selected activities and recursively call the function
    if m < n:
        return [(start[m], finish[m])] + recursive_activity_selector(start, finish, m, n)
    else:
        return []
def greedy_activity_selector(start: List[int], finish: List[int]):
    n = len(start)
    # (1) Group start and finish times for activities
    activities = list(zip(start, finish))
    # (2) Sort (1) in ascending order of finish times
    activities.sort(key=lambda x: x[1])
    # (3) Select the first activity
    selected_activities = [activities[0]]  # Select the first activity
    index_of_last_choose_activity = 0
    # (4) Start from the next activity
    for index_of_next_possible_activity in range(1, n):
        # (4) If the start time of the m.th activity is greater than the finish time of the k.th activity.
        if activities[index_of_next_possible_activity][0] >= activities[index_of_last_choose_activity][1]:
            # (4.1) Take the action
            selected_activities.append(activities[index_of_next_possible_activity])
            # (4.2) Update the last taken action.
            index_of_last_choose_activity = index_of_next_possible_activity
    return selected_activities

# Example usage
start_times = [1, 3, 0, 5, 8, 5]
finish_times = [2, 4, 6, 7, 9, 9]
print("Selected activities via greedy search:")
for activity in greedy_activity_selector(start_times, finish_times):
    print("Start:", activity[0], " Finish:", activity[1])
# Example usage
n = len(start_times)
print("Selected activities via dynamic programming:")
for activity in recursive_activity_selector(start_times, finish_times, 0, n):
    print("Start:", activity[0], " Finish:", activity[1])
