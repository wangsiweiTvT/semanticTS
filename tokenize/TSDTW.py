import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],  # insertion
                                          dtw_matrix[i, j - 1],  # deletion
                                          dtw_matrix[i - 1, j - 1])  # match
    return dtw_matrix[n, m]


def find_top_n_matches(short_seq, long_seq, n):
    len_short = len(short_seq)
    distances = []

    for i in range(len(long_seq) - len_short + 1):
        sub_seq = long_seq[i:i + len_short]
        dist = dtw_distance(short_seq, sub_seq)
        distances.append((dist, i, i + len_short))

    # Sort distances from smallest to largest
    distances.sort()

    # Select top n matches
    top_n_matches = distances[:n]
    return top_n_matches


# Example sequences
short_seq = np.array([1, 2, 3, 2, 1])
long_seq = np.array([0, 1, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1])

# Find top n matches
n = 3
top_n_matches = find_top_n_matches(short_seq, long_seq, n)

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(long_seq, label='Long Sequence')

colors = ['red', 'green', 'blue', 'orange', 'purple']
for idx, (dist, start, end) in enumerate(top_n_matches):
    color = colors[idx % len(colors)]
    plt.plot(range(start, end), long_seq[start:end], label=f'Match {idx + 1} (Dist: {dist:.2f})', color=color)
    plt.scatter(range(start, end), long_seq[start:end], color=color)

plt.title(f'DTW Top {n} Matches')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()
