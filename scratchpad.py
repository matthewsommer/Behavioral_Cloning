import csv
import matplotlib.pyplot as plt

data_csv_file_path = 'data/driving_log.csv'

lines = []
angles = []

with open(data_csv_file_path, newline='') as csvfile:
    csv_data = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in csv_data:
        lines.append(row)

for index, line in enumerate(lines):
    if index == 0:
        print(float(line[3]))
    angles.append(float(line[3]))

plt.hist(angles)
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
