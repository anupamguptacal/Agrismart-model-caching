import random

def get_value_from_datapoint(line):
	return line.split(",")[3]

def get_details_from_datapoint(line):
	if(line.endswith("\n")):
		line = line[:-1]
	a = line.split(",")[2:]
	k = a.pop(1)
	return ','.join(a) + "|" + k

FILE_NAME = "diseases_sanitized.csv"
file = open(FILE_NAME);
file_test = open("temp-test.txt", "w")
current_disease = ""
zero_datapoints = []
non_zero_datapoints = []
start = True
for line in file :
	print(line)
	split_line = line.split(",")
	print(split_line)
	if(start):
		file_test.write(split_line[0] + "\n")
	if(split_line[0] != current_disease):
		if(not start):
			print(len(zero_datapoints))
			print(len(non_zero_datapoints))
			val = random.randint(0, len(zero_datapoints) - 1)
			file_test.write(get_details_from_datapoint(zero_datapoints[val]) + "\n")
			val = random.randint(0, len(non_zero_datapoints) - 1)
			file_test.write(get_details_from_datapoint(non_zero_datapoints[val]) + "\n")
			file_test.write("--\n")
			file_test.write(split_line[0] + "\n")

		start = False
		zero_datapoints = []
		non_zero_datapoints = []
		current_disease = split_line[0]

	if(get_value_from_datapoint(line) == '0'):
		zero_datapoints.append(line)
	else:
		non_zero_datapoints.append(line)

val = random.randint(0, len(zero_datapoints) - 1)
file_test.write(get_details_from_datapoint(zero_datapoints[val]) + "\n")
val = random.randint(0, len(non_zero_datapoints) - 1)
file_test.write(get_details_from_datapoint(non_zero_datapoints[val]) + "\n")
file_test.write("--\n")


