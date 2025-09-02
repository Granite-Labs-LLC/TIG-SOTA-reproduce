import random
import csv

NUM_ROWS = 10_000
MAX_INDEX = 999_999

with open("random_indexes.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["query_index", "random_index"])  # header
    for query_idx in range(NUM_ROWS):
        random_index = random.randint(0, MAX_INDEX)
        writer.writerow([query_idx, random_index])

print("Wrote random_indexes.csv")

