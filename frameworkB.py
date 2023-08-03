from utils import *
import os
import random
import math
import math
import numpy as np
import json
import os
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

class frameworkB:

	def __init__(self,
				 FILE_LOCATION_DISEASE_MAPPING,
				 DISEASE_SPECIFIC_TESTING_DATA,
				 FIELD_TO_MODEL_MAPPING, 
				 STATE_FIELD_MAPPING):
		start_time = datetime.now()
		self.FILE_LOCATION_DISEASE_MAPPING = FILE_LOCATION_DISEASE_MAPPING;
		self.DISEASE_SPECIFIC_TESTING_DATA = DISEASE_SPECIFIC_TESTING_DATA;
		self.FIELD_TO_MODEL_MAPPING = FIELD_TO_MODEL_MAPPING;
		self.CITY_MODELS = {}
		self.SD_MODELS = {}
		self.D_MODELS = {}
		self.STATE_MODELS = {}
		self.INFERENCE_RESULT = {}
		self.STATE_FIELD_MAPPING = STATE_FIELD_MAPPING;
		state_names = next(os.walk('.'))[1]
		for state in state_names:
			if(state == ".git" or state == ".__pycache__"):
				continue;
			self.CITY_MODELS[state] = {}
			self.SD_MODELS[state] = {}
			self.D_MODELS[state] = {}
			self.STATE_MODELS[state] = {}
			captured_diseases_district = {}
			district_names = next(os.walk('./' + state))[1]
			for district in district_names:
				self.CITY_MODELS[state][district] = {}
				self.SD_MODELS[state][district] = {}
				self.D_MODELS[state][district] = {}
				sub_district_names = next(os.walk('./' + state + "/" + district))[1]
				captured_diseases_sub_district = {}
				for sub_district in sub_district_names:
					self.CITY_MODELS[state][district][sub_district] = {}
					self.SD_MODELS[state][district][sub_district] = {}
					node_names = next(os.walk('./' + state + "/" + district + "/" + sub_district))[1]
					captured_diseases_city = {}

					for city in node_names:
						self.CITY_MODELS[state][district][sub_district][city] = {}
						fields = next(os.walk('./' + state + '/' + district + "/" + sub_district + "/" + city))[1]
						captured_diseases = {}
						for i in range(len(fields)):
							field = fields[i]
							file_location = construct_compiled_file_location(state, district, sub_district, city, field)
							disease_set = self.FILE_LOCATION_DISEASE_MAPPING[file_location]
							for disease in disease_set:
								if(disease not in captured_diseases):
									captured_diseases[disease] = [i]
								else:
									captured_diseases[disease].append(i)

						for disease in captured_diseases:
							field_indices = captured_diseases[disease]
							random_value = random.randint(0, 4)
							random_datapoint_for_disease = self.DISEASE_SPECIFIC_TESTING_DATA[disease][random_value]
							min_error = 9999999999999999999999999.0
							selected_model = None
							index = None
							for index in field_indices:
								field = fields[index]
								models_of_field = self.FIELD_TO_MODEL_MAPPING[state][district][sub_district][city][field]

								model_for_disease = models_of_field[disease]
								x_value = np.expand_dims(random_datapoint_for_disease[0], axis=0)
								inference = model_for_disease.predict(x_value, verbose=0)
								inference = inference[0][0]

								MSE = np.square(np.subtract([inference],[random_datapoint_for_disease[1]])).mean()
								MSE = MSE if inference < 0.0 else MSE * -1.0
								if(MSE < min_error):
									min_error = MSE
									selected_model = model_for_disease
							self.CITY_MODELS[state][district][sub_district][city][disease] = selected_model

							if(disease not in captured_diseases_city):
								captured_diseases_city[disease] = [city]
							else:
								captured_diseases_city[disease].append(city) 

					for disease in captured_diseases_city:
						city_indices = captured_diseases_city[disease]
						min_error = 99999999999999999.0
						selected_model = None
						random_value = random.randint(0, 4)
						random_datapoint_for_disease = self.DISEASE_SPECIFIC_TESTING_DATA[disease][random_value]
						city = None
						for city in city_indices:
							model_for_city = self.CITY_MODELS[state][district][sub_district][city][disease]

							inference = model_for_city.predict(np.expand_dims(random_datapoint_for_disease[0], axis=0), verbose=0)[0][0]

							MSE = np.square(np.subtract([inference],[random_datapoint_for_disease[1]])).mean()
							MSE = MSE if inference < 0.0 else MSE * -1.0
							if(MSE < min_error):
								min_error = MSE
								selected_model = model_for_city

						self.SD_MODELS[state][district][sub_district][disease] = selected_model

						if(disease not in captured_diseases_sub_district):
							captured_diseases_sub_district[disease] = [sub_district]
						else:
							captured_diseases_sub_district[disease].append(sub_district) 

				for disease in captured_diseases_sub_district:
					sd_indices = captured_diseases_sub_district[disease]
					min_error = 99999999999999999.0
					selected_model = None
					random_value = random.randint(0, 4)
					random_datapoint_for_disease = self.DISEASE_SPECIFIC_TESTING_DATA[disease][random_value]
					for sd in sd_indices:
						model_for_sd = self.SD_MODELS[state][district][sd][disease]

						inference = model_for_sd.predict(np.expand_dims(random_datapoint_for_disease[0], axis=0), verbose=0)[0][0]

						MSE = np.square(np.subtract([inference],[random_datapoint_for_disease[1]])).mean()
						MSE = MSE if inference < 0.0 else MSE * -1.0
						if(MSE < min_error):
							min_error = MSE
							selected_model = model_for_sd

					self.D_MODELS[state][district][disease] = selected_model

					if(disease not in captured_diseases_district):
						captured_diseases_district[disease] = [district]
					else:
						captured_diseases_district[disease].append(district) 

			for disease in captured_diseases_district:
				d_indices = captured_diseases_district[disease]
				min_error = 99999999999999999.0
				selected_model = None
				random_value = random.randint(0, 4)
				random_datapoint_for_disease = self.DISEASE_SPECIFIC_TESTING_DATA[disease][random_value]
				for d in d_indices:
					model_for_d = self.D_MODELS[state][d][disease]

					inference = model_for_d.predict(np.expand_dims(random_datapoint_for_disease[0], axis=0), verbose=0)[0][0]

					MSE = np.square(np.subtract([inference],[random_datapoint_for_disease[1]])).mean()
					MSE = MSE if inference < 0.0 else MSE * -1.0
					if(MSE < min_error):
						min_error = MSE
						selected_model = model_for_d

				self.STATE_MODELS[state][disease] = selected_model
		end_time = datetime.now()
		print("Initialization time : ", str((end_time - start_time).total_seconds()))

	def infer(self, test_data, tier_of_choice):
		self.INFERENCE_RESULT = {}
		print_separation_line()

		state_names = next(os.walk('.'))[1]
		if(tier_of_choice == 'field'):
			for state in state_names:
				if(state == '.git' or state == '__pycache__'):
					continue;
				self.INFERENCE_RESULT[state] = {}
				district_names = next(os.walk('./' + state))[1]
				for district in district_names:
					self.INFERENCE_RESULT[state][district] = {}
					sub_district_names = next(os.walk('./' + state + "/" + district))[1]
					for sub_district in sub_district_names:
						self.INFERENCE_RESULT[state][district][sub_district] = {}
						node_names = next(os.walk('./' + state + "/" + district + "/" + sub_district))[1]
						for city in node_names:
							fields = next(os.walk('./' + state + '/' + district + "/" + sub_district + "/" + city))[1]
							self.INFERENCE_RESULT[state][district][sub_district][city] = {}
							for field in fields:
								self.INFERENCE_RESULT[state][district][sub_district][city][field] = {}
								models_of_field = self.FIELD_TO_MODEL_MAPPING[state][district][sub_district][city][field]
								disease_set = self.STATE_FIELD_MAPPING[state][district][sub_district][city][field]

								for i in range(len(disease_set)):
									model_of_disease = models_of_field[disease_set[i]]

									self.INFERENCE_RESULT[state][district][sub_district][city][field][disease_set[i]] = model_of_disease.predict(np.expand_dims(test_data, axis=0), verbose = 0)[0][0]

			return self.INFERENCE_RESULT

		elif(tier_of_choice == 'city'):
			for state in state_names:
				if(state == ".git"):
					continue;
				self.INFERENCE_RESULT[state] = {}
				district_names = next(os.walk('./' + state))[1]
				for district in district_names:
					self.INFERENCE_RESULT[state][district] = {}
					sub_district_names = next(os.walk('./' + state + "/" + district))[1]
					for sub_district in sub_district_names:
						self.INFERENCE_RESULT[state][district][sub_district] = {}
						node_names = next(os.walk('./' + state + "/" + district + "/" + sub_district))[1]
						for city in node_names:
							self.INFERENCE_RESULT[state][district][sub_district][city]= {}
							for disease in self.CITY_MODELS[state][district][sub_district][city]:
								self.INFERENCE_RESULT[state][district][sub_district][city][disease] = self.CITY_MODELS[state][district][sub_district][city][disease].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]

			return self.INFERENCE_RESULT

		elif(tier_of_choice == 'sub_district'):
			for state in state_names:
				if(state == ".git"):
					continue;
				self.INFERENCE_RESULT[state] = {}
				district_names = next(os.walk('./' + state))[1]
				for district in district_names:
					self.INFERENCE_RESULT[state][district] = {}
					sub_district_names = next(os.walk('./' + state + "/" + district))[1]
					for sub_district in sub_district_names:
						self.INFERENCE_RESULT[state][district][sub_district] = {}

						for disease in self.SD_MODELS[state][district][sub_district]:
							self.INFERENCE_RESULT[state][district][sub_district][disease] = self.SD_MODELS[state][district][sub_district][disease].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]

			return self.INFERENCE_RESULT

		elif(tier_of_choice == 'district'):
			for state in state_names:
				if(state == ".git"):
					continue;
				self.INFERENCE_RESULT[state] = {}
				district_names = next(os.walk('./' + state))[1]
				for district in district_names:
					self.INFERENCE_RESULT[state][district] = {}

					for disease in self.D_MODELS[state][district]:
						self.INFERENCE_RESULT[state][district][disease] = self.D_MODELS[state][district][disease].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]

			return self.INFERENCE_RESULT

		elif(tier_of_choice == 'state'):
			for state in state_names:
				if(state == ".git"):
					continue;
				self.INFERENCE_RESULT[state] = {}
				district_names = next(os.walk('./' + state))[1]
			
				for disease in self.STATE_MODELS[state]:
					self.INFERENCE_RESULT[state][disease] = self.STATE_MODELS[state][disease].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]

			return self.INFERENCE_RESULT

		else:
			return [-1]

