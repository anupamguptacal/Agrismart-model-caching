import os
import random
import math
import math
import numpy as np
import json
import os
import tensorflow as tf
from tensorflow import keras
from frameworkB import *
from frameworkA import *
from frameworkC import *
from datetime import datetime

STATE_FIELD_MAPPING = {}
STATE_MODEL_MAPPING = {}
WRITE_FILE_NAME = "/diseases_sanitized.csv"
TRAINING_X_FILE_EXTENSION = "_train_x.csv"
TRAINING_Y_FILE_EXTENSION = "_train_y.csv"
FIELD_DATA_FILE_LOCATIONS = []
FILE_LOCATION_DISEASE_MAPPING = {}
TRAIN_Y_MAPPING = {}
OPTIMIZER_Y_MAPPING = {}
FIELD_TO_MODEL_MAPPING = {}
DISEASE_SPECIFIC_TESTING_DATA = {}
GLOBAL_EPOCHS = 50
GLOBAL_BATCH_SIZE = 32
DATASET_FIELD_STRENGTH = {}


def pre_processing():
	print_separation_line()
	state_names = next(os.walk('.'))[1]

	for state in state_names:
		if(state == '.git' or state == '__pycache__'):
			continue;
		dictionary_state_field_level = {}
		dictionary_state_model_level = {}
		dictionary_y_train_state_level = {}
		dictionary_state_optimizer_level = {}
		DATASET_FIELD_STRENGTH[state] = {}
		district_names = next(os.walk('./' + state))[1]
		for district in district_names:
			dictionary_district_field_level = {}
			dictionary_district_model_level = {}
			dictionary_y_train_district_level = {}
			dictionary_district_optimizer_level = {}
			DATASET_FIELD_STRENGTH[state][district] = {}

			sub_district_names = next(os.walk('./' + state + "/" + district))[1]

			for sub_district in sub_district_names:
				dictionary_sub_district_field_level = {}
				dictionary_sub_district_model_level = {}
				dictionary_y_train_sub_district_level = {}
				dictionary_sub_district_optimizer_level = {}
				DATASET_FIELD_STRENGTH[state][district][sub_district] = {}
				node_names = next(os.walk('./' + state + "/" + district + "/" + sub_district))[1]
				
				for city in node_names:
					dictionary_field = {}
					dictionary_model = {}
					dictionary_y_mapping = {}
					dictionary_optimizer = {}
					DATASET_FIELD_STRENGTH[state][district][sub_district][city] = {}

					fields = next(os.walk('./' + state + '/' + district + "/" + sub_district + "/" + city))[1]
					for field in fields:
						DATASET_FIELD_STRENGTH[state][district][sub_district][city][field] = {}
						write_file_name = construct_compiled_file_location(state, district, sub_district, city, field)
						FIELD_DATA_FILE_LOCATIONS.append(write_file_name)

						write_file = open(write_file_name, "w")
						disease_names = next(os.walk('./' + state + '/' + district + "/" + sub_district + "/" + city + '/' + field))[1]
						disease_set = set()
						for disease_name in disease_names:
							file_name = os.listdir('./' + state + '/' + district + "/" + sub_district + "/" + city + '/' + field + '/' + disease_name)[0]
							file_location = './' + state + '/' + district + "/" + sub_district + "/" + city + '/' + field + '/' + disease_name + '/' + file_name
							datapoints = open(file_location)
							disease_set.add(disease_name)
							for line in datapoints:
								write_file.write(disease_name + "," + line)
								if(not line.endswith("\n")):
									write_file.write("\n")
							print("Data Intake and Pre-Processing for " + file_location + " completed.")

						FILE_LOCATION_DISEASE_MAPPING[write_file_name] = list(disease_set)
						dictionary_field[field] = list(disease_set)
						dictionary_model[field] = None
						dictionary_optimizer[field] = None

						count = 0
						mapping = {}
						for disease in disease_set:
							mapping[disease] = count
							count = count + 1
						dictionary_y_mapping[field] = mapping
					dictionary_sub_district_field_level[city] =  dictionary_field
					dictionary_sub_district_model_level[city] = dictionary_model
					dictionary_y_train_sub_district_level[city] = dictionary_y_mapping
					dictionary_sub_district_optimizer_level[city] = dictionary_optimizer

				dictionary_district_field_level[sub_district] =  dictionary_sub_district_field_level
				dictionary_district_model_level[sub_district] = dictionary_sub_district_model_level
				dictionary_y_train_district_level[sub_district] = dictionary_y_train_sub_district_level
				dictionary_district_optimizer_level[sub_district] = dictionary_sub_district_optimizer_level

			dictionary_state_field_level[district] =  dictionary_district_field_level
			dictionary_state_model_level[district] = dictionary_district_model_level
			dictionary_y_train_state_level[district] = dictionary_y_train_district_level
			dictionary_state_optimizer_level[district] = dictionary_district_optimizer_level

		TRAIN_Y_MAPPING[state] = dictionary_y_train_state_level
		OPTIMIZER_Y_MAPPING[state] = dictionary_state_optimizer_level
		STATE_FIELD_MAPPING[state] = dictionary_state_field_level
		STATE_MODEL_MAPPING[state] = dictionary_state_model_level

def data_sanitization():
	print_separation_line()
	for write_file_name in FIELD_DATA_FILE_LOCATIONS:
		temp_file_location = write_file_name[:-4] + "_temp.csv"
		read_file = open(write_file_name)
		write_file = open(temp_file_location, "w")
		for line in read_file:
			delimites = line.split(",")
			changed = []
			for character in delimites:
				if len(character) == 0:
					changed.append('0')
				else:
					changed.append(character)
			if len(changed) != 12:
				for r in range(12 - len(changed) + 1):
					changed.append('0')
			new_line = ",".join(changed)
			write_file.write(new_line)

		write_file = open(write_file_name, "w")
		read_file = open(temp_file_location)
		for line in read_file:
			write_file.write(line)
		read_file.close()
		write_file.close()
		os.remove(temp_file_location)

		print("Data Santization for " + write_file_name + " completed.")

def model_creation():
	print_separation_line()
	state_names = next(os.walk('.'))[1]
	for state in state_names:
		if(state == '.git' or state == '__pycache__'):
			continue;
		district_names = next(os.walk('./' + state))[1]
		for district in district_names:
			sub_district_names = next(os.walk('./' + state + "/" + district))[1]
			for sub_district in sub_district_names:
				node_names = next(os.walk('./' + state + "/" + district + "/" + sub_district))[1]
				for city in node_names:
					fields = next(os.walk('./' + state + '/' + district + "/" + sub_district + "/" + city))[1]
					for field in fields:
						file_location = construct_compiled_file_location(state, district, sub_district, city, field)
						disease_set = FILE_LOCATION_DISEASE_MAPPING[file_location]
						models = create_classification_model(len(disease_set))
						STATE_MODEL_MAPPING[state][district][sub_district][city][field] = models

						print("DNN Model for " + field + " in city " + city + " in sub_district " + sub_district + " in district: " + district + " in state " + state + " constructed")


def create_classification_model(disease_count):
	models = []
	optimizers = []
	for x in range(disease_count):
		model = tf.keras.models.Sequential()
		model.add(tf.keras.Input(shape = (9, )))
		model.add(tf.keras.layers.Dense(32))
		model.add(tf.keras.layers.Dense(1))
		model.compile(optimizer='adam',
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['accuracy'])
		models.append(model)
		#model.summary()

	return models

def partition_data_set():
	print_separation_line()
	for write_file_name in FIELD_DATA_FILE_LOCATIONS:
		read_file = open(write_file_name)

		training_file_name = write_file_name[:-4] + TRAINING_X_FILE_EXTENSION
		test_file_name = write_file_name[:-4] + TRAINING_Y_FILE_EXTENSION


		training_file = open(training_file_name, "w")
		test_file = open(test_file_name, "w")

		for line in read_file:
			split_line = line.split(",")
			test_file.write(split_line[3] + "," + split_line[0] + "\n")
			# year in data is dropped
			training_file.write(split_line[2] + "," + ",".join(split_line[4:]))

		read_file.close()
		training_file.close()
		test_file.close()

		print("Training and test data for " + write_file_name + " created. ")

def train_models():
	print_separation_line()
	state_names = next(os.walk('.'))[1]
	for state in state_names:
		if(state == '.git' or state == '__pycache__'):
			continue;
		district_names = next(os.walk('./' + state))[1]
		for district in district_names:
			sub_district_names = next(os.walk('./' + state + "/" + district))[1]
			for sub_district in sub_district_names:
				node_names = next(os.walk('./' + state + "/" + district + "/" + sub_district))[1]
				for city in node_names:
					fields = next(os.walk('./' + state + '/' + district + "/" + sub_district + "/" + city))[1]
					for field in fields:
						print("Training model for state: " + state + ", city: " + city + ", field = " + field)
						x_train_location = construct_train_x_file_location(state, district, sub_district, city, field)
						y_train_location = construct_train_y_file_location(state, district, sub_district, city, field)

						x_train_file = open(x_train_location)
						y_train_file = open(y_train_location)


						#create encoding for y_train
						disease_set = STATE_FIELD_MAPPING[state][district][sub_district][city][field]
						#print(disease_set)

						disease_wise_separation_x = []
						disease_wise_separation_y = []
						for i in range(len(disease_set)):
							disease_wise_separation_x.append([])
							disease_wise_separation_y.append([])

						mapping_dictionary = TRAIN_Y_MAPPING[state][district][sub_district][city][field]
						with open(x_train_location) as file1, open(y_train_location) as file2:
							for linex, liney in zip(file1, file2):
								#modify x datapoint
								split = linex.split(",")
								last_element = split[len(split) - 1]
								if last_element.endswith("\n"):
									split.pop()
									split.append(last_element[:-1])
								float_converted = [float(num) for num in split]

								if liney.endswith("\n"):
									liney = liney[:-1]

								disease_name = liney.split(",")[1]
								disease_index = mapping_dictionary[liney.split(",")[1]]
								disease_value = float(liney.split(",")[0])

								if(disease_name not in DISEASE_SPECIFIC_TESTING_DATA) :
									to_add = [float_converted, disease_value]
									DISEASE_SPECIFIC_TESTING_DATA[disease_name] = [to_add]
								elif(len(DISEASE_SPECIFIC_TESTING_DATA[disease_name]) < 5):
									DISEASE_SPECIFIC_TESTING_DATA[disease_name].append([float_converted, disease_value])
								else:
									disease_wise_separation_x[disease_index].append(float_converted)
									disease_wise_separation_y[disease_index].append(disease_value)

						models = get_model_for_field(state, district, sub_district, city, field)
						mapping_dict_inv = {v: k for k, v in mapping_dictionary.items()}

						if(state not in FIELD_TO_MODEL_MAPPING):
							FIELD_TO_MODEL_MAPPING[state] = {}
						if(district not in FIELD_TO_MODEL_MAPPING[state]):
							FIELD_TO_MODEL_MAPPING[state][district] = {}
						if(sub_district not in FIELD_TO_MODEL_MAPPING[state][district]):
							FIELD_TO_MODEL_MAPPING[state][district][sub_district] = {}
						if(city not in FIELD_TO_MODEL_MAPPING[state][district][sub_district]) :
							FIELD_TO_MODEL_MAPPING[state][district][sub_district][city] = {}
						if(field not in FIELD_TO_MODEL_MAPPING[state][district][sub_district][city]):
							FIELD_TO_MODEL_MAPPING[state][district][sub_district][city][field] = {}


						for i in range(len(disease_set)):
							print("Training for disease: ", mapping_dict_inv[i], " for state: ", str(state), ", city: ", str(city), ", field: ", str(field));
							model_of_disease = models[i]
							FIELD_TO_MODEL_MAPPING[state][district][sub_district][city][field][mapping_dict_inv[i]] = models[i]
							DATASET_FIELD_STRENGTH[state][district][sub_district][city][field][mapping_dict_inv[i]] = len(disease_wise_separation_x[i])
							x_values = disease_wise_separation_x[i]
							y_values = disease_wise_separation_y[i]
							#model_of_disease.fit(x_values, y_values, epochs = 150, verbose=0)

def get_model_for_field(state, district, sub_district, city, field):
	return STATE_MODEL_MAPPING[state][district][sub_district][city][field]

def test_models():
	print_separation_line()
	print("Running Frame specific tests for City level");
	state_names = next(os.walk('.'))[1]
	#framework = input("Please enter a framework for computation (a/b/c): ")
	frame = None
	localization_weightage = 0.1
	communication_overhead_weightage = 2.0
	avoid_concurrent_tiers_weightage = 5.0
	frameworks = [ frameworkC(1, FILE_LOCATION_DISEASE_MAPPING, DISEASE_SPECIFIC_TESTING_DATA, FIELD_TO_MODEL_MAPPING, DATASET_FIELD_STRENGTH, communication_overhead_weightage, avoid_concurrent_tiers_weightage, localization_weightage), frameworkA(FIELD_TO_MODEL_MAPPING, DATASET_FIELD_STRENGTH), frameworkB(FILE_LOCATION_DISEASE_MAPPING, DISEASE_SPECIFIC_TESTING_DATA, FIELD_TO_MODEL_MAPPING, STATE_FIELD_MAPPING)]
	#if(framework == 'a' or framework == 'A'):
	#	frame = frameworkA(FIELD_TO_MODEL_MAPPING)
	#elif(framework == 'b' or framework == 'B'):
	#	frame = frameworkB(FILE_LOCATION_DISEASE_MAPPING, DISEASE_SPECIFIC_TESTING_DATA, FIELD_TO_MODEL_MAPPING, STATE_FIELD_MAPPING)
	#elif(framework == 'c' or framework == 'C'):
	#	localization_weightage = 0.1
	#	communication_overhead_weightage = 2.0
	#	avoid_concurrent_tiers_weightage = 5.0
	#	frame = frameworkC(2, FILE_LOCATION_DISEASE_MAPPING, DISEASE_SPECIFIC_TESTING_DATA, FIELD_TO_MODEL_MAPPING, communication_overhead_weightage, avoid_concurrent_tiers_weightage, localization_weightage)



	for frame in frameworks:
		mse_values = []
		for state in state_names:
			if(state == '.git' or state == '__pycache__'):
				continue;
			district_names = next(os.walk('./' + state))[1]
			for district in district_names:
				sub_district_names = next(os.walk('./' + state + "/" + district))[1]
				for sub_district in sub_district_names:
					node_names = next(os.walk('./' + state + "/" + district + "/" + sub_district))[1]
					for city in node_names:
						fields = next(os.walk('./' + state + '/' + district + "/" + sub_district + "/" + city))[1]
						for field in fields:
							#field_val_1 = frame.infer(float_converted, 'field')
							#ity_val_1 = frame.infer(float_converted, 'city')
							#sub_district_val_1 = frame.infer(float_converted, 'sub_district')
							#district_val_1 = frame.infer(float_converted, 'district')
							#state_val_1 = frame.infer(float_converted, 'state')


							file_location =  './' + state + '/' + district + '/' + sub_district + '/' + city + '/' + field + '/test-data.txt'
							test_metrics = {}
							file = open(file_location).readlines()
							counter = 0;
							while counter < len(file):
								if(file[counter] == "\n"):
									continue;
								disease_name = file[counter][:-1]
								test_metrics[disease_name] = []
								counter = counter + 1
								line = file[counter][:-1]
								test_metrics[disease_name].append(line)
								counter = counter + 1
								line = file[counter][:-1]
								test_metrics[disease_name].append(line)
								counter = counter + 2
							
							diseases = FIELD_TO_MODEL_MAPPING[state][district][sub_district][city][field]
							for disease in diseases:
								#print_separation_line()
								print("Testing for disease: ", disease, " for field: ", field, ", city: ", city, " for framework: ", frame)
								model = diseases[disease]
								test_data = test_metrics[disease]
								for test in test_data:
									a = test.split(",")
									last_datapoint = float(a[-1].split('|')[0])
									val = float(a[-1].split('|')[1])
									a.pop(len(a) - 1)
									a.append(last_datapoint)
									float_converted = [float(num) for num in a]

									field_val = frame.infer(float_converted, 'sub_district')
									field_val = field_val[state][district][sub_district][disease]

									if type(field_val) == list:
										mse_values.append(np.square(np.subtract([val],[field_val[1]])).mean())
									else:
										mse_values.append(np.square(np.subtract([val],[field_val])).mean())

									#city_val = city_val_1[state][district][sub_district][city][disease]
									#print("Testing city")
									#if type(field_val) == list:
									#	mse_values.append(np.square(np.subtract([val],[city_val[1]])).mean())
									#else:
									#	mse_values.append(np.square(np.subtract([val],[city_val])).mean())

									#sub_district_val = sub_district_val_1[state][district][sub_district][disease]
									#print("Testing Sub District")
									#if type(field_val) == list:
									#	mse_values.append(np.square(np.subtract([val],[sub_district_val[1]])).mean())
									#else:
									#	mse_values.append(np.square(np.subtract([val],[sub_district_val])).mean())

									#district_val = district_val_1[state][district][disease]
									#print("Testing District")
									#if type(field_val) == list:
									#	mse_values.append(np.square(np.subtract([val],[district_val[1]])).mean())
									#else:
									#	mse_values.append(np.square(np.subtract([val],[district_val])).mean())

									#state_val = state_val_1[state][disease]
									#print("Testing State")
									#if type(field_val) == list:
									#	mse_values.append(np.square(np.subtract([val],[state_val[1]])).mean())
									#else:
									#	mse_values.append(np.square(np.subtract([val],[state_val])).mean())
									#print_separation_line()
									#break;
		print("For framework: ", frame, ", mse = ", mse_values)
		print("Average: ", sum(mse_values)/len(mse_values))
								
def main():
	pre_processing()
	data_sanitization()
	model_creation()
	partition_data_set()
	train_models()
	"""
	data = "16,40.3,23.6,43.9,26.7,0.0,2.3,6.2,5.8"
	split_data = data.split(",")
	data2 = []
	for k in split_data:
		data2.append(float(k))

	
	tier = "state"

	tiers = ["state", "district", "sub_district", "city", "field"]
	split_data = data.split(",")
	data2 = []
	for k in split_data:
		data2.append(float(k))

	for tier in tiers:
		average = 0.0
		#a = frameworkA(FIELD_TO_MODEL_MAPPING, DATASET_FIELD_STRENGTH)
		#a = frameworkB(FILE_LOCATION_DISEASE_MAPPING, DISEASE_SPECIFIC_TESTING_DATA, FIELD_TO_MODEL_MAPPING, STATE_FIELD_MAPPING)

		localization_weightage = 0.1
		communication_overhead_weightage = 2.0
		avoid_concurrent_tiers_weightage = 5.0
		a = frameworkC(1, FILE_LOCATION_DISEASE_MAPPING, DISEASE_SPECIFIC_TESTING_DATA, FIELD_TO_MODEL_MAPPING, DATASET_FIELD_STRENGTH, communication_overhead_weightage, avoid_concurrent_tiers_weightage, localization_weightage)

		for i in range(10):
			start_time = datetime.now()
			val = a.infer(data2, tier)
			end_time = datetime.now()
			#print(val)
			time = (end_time - start_time).total_seconds()
			average = average + time
			print("Evaluation Time : ", str(time))

		average = average/10.0

		print("For tier ", tier, " average = ", str(average))

		print_separation_line()
	
	a = frameworkB(FILE_LOCATION_DISEASE_MAPPING, DISEASE_SPECIFIC_TESTING_DATA, FIELD_TO_MODEL_MAPPING, STATE_FIELD_MAPPING)
	for i in range(10):
		start_time = datetime.now()
		val = a.infer(data2, tier)
		end_time = datetime.now()
		print("Evaluation Time : ", str(-1.0 * (start_time - end_time).total_seconds()))

	print_separation_line()

	localization_weightage = 0.1
	communication_overhead_weightage = 2.0
	avoid_concurrent_tiers_weightage = 5.0
	a = frameworkC(1, FILE_LOCATION_DISEASE_MAPPING, DISEASE_SPECIFIC_TESTING_DATA, FIELD_TO_MODEL_MAPPING, DATASET_FIELD_STRENGTH, communication_overhead_weightage, avoid_concurrent_tiers_weightage, localization_weightage)
	for i in range(10):
		start_time = datetime.now()
		val = a.infer(data2, tier)
		end_time = datetime.now()
		print("Evaluation Time : ", str(-1.0 * (start_time - end_time).total_seconds()))

	print_separation_line()
	

	"""
	framework = input("Please enter a framework for computation (a/b/c): ")
	a = None
	if(framework == 'a' or framework == 'A'):
		a = frameworkA(FIELD_TO_MODEL_MAPPING, DATASET_FIELD_STRENGTH)
	elif(framework == 'b' or framework == 'B'):
		a = frameworkB(FILE_LOCATION_DISEASE_MAPPING, DISEASE_SPECIFIC_TESTING_DATA, FIELD_TO_MODEL_MAPPING, STATE_FIELD_MAPPING)
	elif(framework == 'c' or framework == 'C'):
		localization_weightage = 0.1
		communication_overhead_weightage = 2.0
		avoid_concurrent_tiers_weightage = 5.0
		a = frameworkC(1, FILE_LOCATION_DISEASE_MAPPING, DISEASE_SPECIFIC_TESTING_DATA, FIELD_TO_MODEL_MAPPING, DATASET_FIELD_STRENGTH, communication_overhead_weightage, avoid_concurrent_tiers_weightage, localization_weightage)
	else:
		print("Unrecognized framework");
		return;
	#print(a.infer(data2, "state"))
	
	while(True):
		data = input("Please enter the test-data value as a comma separated 9 value string")#: (enter to use default)")
		#data = "16,40.3,23.6,43.9,26.7,0.0,2.3,6.2,5.8"

		split_data = data.split(",")
		data2 = []
		for k in split_data:
			data2.append(float(k))

		tier = input("Please enter a tier of interest - 'field', 'city', 'sub_district', 'district', 'state' : ")
		#[16,40.3,23.6,43.9,26.7,0.0,2.3,6.2,5.8], 'field')
		#start_time = datetime.now()
		result = a.infer(data2, tier)
		#end_time = datetime.now()
		#print("Inference Time : ", str((end_time - start_time).total_seconds()))
		
		print_separation_line()
		print("Results: ")
		print(result)
		print_separation_line()
	"""
	#test_models()
	
if __name__ == "__main__":
	main()
