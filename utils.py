import os
import random
import math
import math
import numpy as np
import json
import os
import tensorflow as tf
from tensorflow import keras

WRITE_FILE_NAME = "/diseases_sanitized.csv"
TRAINING_X_FILE_EXTENSION = "_train_x.csv"
TRAINING_Y_FILE_EXTENSION = "_train_y.csv"	

def extract_value_from_cache(tier_of_choice, STATE_MODELS, D_MODELS, SD_MODELS, CITY_MODELS):
	INFERENCE_RESULT = {}
	for state in state_names:
		INFERENCE_RESULT[state] = {}
		if(tier_of_choice == "state"):
			for disease in STATE_MODELS[state]:
				INFERENCE_RESULT[state][disease] = STATE_MODELS[state][disease].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]
			continue;
		district_names = next(os.walk('./' + state))[1]
		for district in district_names:
			INFERENCE_RESULT[state][district] = {}
			if(tier_of_choice == "district"): 
				for disease in D_MODELS[state][district]:
					INFERENCE_RESULT[state][district][disease] = D_MODELS[state][district][disease].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]	
				continue;

			sub_district_names = next(os.walk('./' + state + "/" + district))[1]
			for sub_district in sub_district_names:
				INFERENCE_RESULT[state][district][sub_district] = {}
				if(tier_of_choice == "sub_district"):
					for disease in SD_MODELS[state][district][sub_district]:
						INFERENCE_RESULT[state][district][sub_district][disease] = SD_MODELS[state][district][sub_district][disease].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]
					continue;
				node_names = next(os.walk('./' + state + "/" + district + "/" + sub_district))[1]
				for city in node_names:
					INFERENCE_RESULT[state][district][sub_district][city]= {}
					for disease in CITY_MODELS[state][district][sub_district][city]:
						INFERENCE_RESULT[state][district][sub_district][city][disease] = CITY_MODELS[state][district][sub_district][city][disease].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]		
	return INFERENCE_RESULT
	
def compute_weighted_inference(values) :#[{disease_name: [support, value]}]
	diseases = {}
	for i in range(len(values)):
		value = values[i]
		for key in value.keys():
			if(key not in diseases):
				diseases[key] = []
			diseases[key].append(i)

	result = {}
	for disease in diseases:
		indices = diseases[disease]
		total_weight = 0.0
		total_sum = 0
		for index in indices:
			disease_dict = values[index]

			support_weights = disease_dict[disease][0]
			weighted_value = disease_dict[disease][1]

			total_sum = total_sum + support_weights
			total_weight = total_weight + weighted_value * support_weights

		result[disease] = [total_sum, (total_weight/total_sum * 1.0)]

	return result;

def construct_compiled_file_location(state, district, sub_district, city, field):
	return './' + state + '/' + district + '/' + sub_district + '/' + city + '/' + field + WRITE_FILE_NAME

def construct_train_x_file_location(state, district, sub_district, city, field):
	return construct_compiled_file_location(state, district, sub_district, city, field)[:-4] + TRAINING_X_FILE_EXTENSION

def construct_train_y_file_location(state, district, sub_district, city, field):
	return construct_compiled_file_location(state, district, sub_district, city, field)[:-4] + TRAINING_Y_FILE_EXTENSION

def print_separation_line():
	print("***************************************************************************")

def place(K, N, minWeight, placements, upperWeight, minPlacement, lowerTotalWeight, communication_overhead_weightage, avoid_concurrent_tiers_weightage, weight):
  currentWeight = upperWeight + lowerTotalWeight
  if(len(placements) > K):
    return -1.0, [1];
  elif(currentWeight > minWeight) :
    return -2.0, [1]
  elif(len(placements) == K):
    if(currentWeight < minWeight) :
      minWeight = currentWeight
      return minWeight, placements;	
    return -3.0, [1]
  maxtier = -1
  if(len(placements) > 0):
    maxtier = placements[len(placements) - 1]

  for i in range(maxtier + 1, N) :
    tiersAbove = (i - (maxtier + 1)) * communication_overhead_weightage
    if(maxtier != -1 and maxtier - i == -1):
      lowerWeightTotal = lowerTotalWeight + tiersAbove + weight[i] + avoid_concurrent_tiers_weightage
    else:
      lowerWeightTotal = lowerTotalWeight + tiersAbove + weight[i]
    upperWeight = (N - (i + 1)) * communication_overhead_weightage
    placements.append(i)
    minWeightCalc, minPlacementCalc = place(K, N, minWeight, placements, upperWeight, minPlacement,lowerWeightTotal, communication_overhead_weightage, avoid_concurrent_tiers_weightage, weight)
    if(minWeightCalc > 0.0):
      minWeight = minWeightCalc
      minPlacement = minPlacementCalc.copy()
    placements.pop()
  return minWeight, minPlacement
