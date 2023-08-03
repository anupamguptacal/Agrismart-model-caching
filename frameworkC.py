from utils import *
from frameworkA import *
from datetime import datetime

class frameworkC:

	def __init__(self,
				 number_of_caches,
				 FILE_LOCATION_DISEASE_MAPPING,
				 DISEASE_SPECIFIC_TESTING_DATA,
				 FIELD_TO_MODEL_MAPPING, 
				 DATASET_FIELD_STRENGTH,
				 communication_overhead_weightage,
				 avoid_concurrent_tiers_weightage,
				 localization_weightage):
		start_time = datetime.now()
		self.number_of_caches = number_of_caches;
		self.FILE_LOCATION_DISEASE_MAPPING = FILE_LOCATION_DISEASE_MAPPING;
		self.DISEASE_SPECIFIC_TESTING_DATA = DISEASE_SPECIFIC_TESTING_DATA;
		self.FIELD_TO_MODEL_MAPPING = FIELD_TO_MODEL_MAPPING;
		self.avoid_concurrent_tiers_weightage = avoid_concurrent_tiers_weightage;
		self.communication_overhead_weightage = communication_overhead_weightage;
		self.localization_weightage = localization_weightage;
		self.CITY_MODELS = {}
		self.D_MODELS = {}
		self.SD_MODELS = {}
		self.STATE_MODELS = {}
		self.tiers_cached = []
		self.DATASET_FIELD_STRENGTH = DATASET_FIELD_STRENGTH
		self.framework_a = frameworkA(FIELD_TO_MODEL_MAPPING, DATASET_FIELD_STRENGTH)
		count_city = 0
		count_sd = 0
		count_d = 0
		count_state = 0
		print("Computing Cache across tiers");
		state_names = next(os.walk('.'))[1]
		for state in state_names:
			if(state == '.git' or state == '__pycache__'):
				continue;			
			self.CITY_MODELS[state] = {}
			self.D_MODELS[state] = {}
			self.SD_MODELS[state] = {}
			self.STATE_MODELS[state] = {}
			captured_diseases_district = {}
			district_names = next(os.walk('./' + state))[1]
			for district in district_names:
				self.CITY_MODELS[state][district] = {}
				self.D_MODELS[state][district] = {}
				self.SD_MODELS[state][district] = {}
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
							selected_model_strength = 0.0
							index = None
							for index in field_indices:
								field = fields[index]
								models_of_field = self.FIELD_TO_MODEL_MAPPING[state][district][sub_district][city][field]

								model_for_disease = models_of_field[disease]
								model_for_disease_strength = self.DATASET_FIELD_STRENGTH[state][district][sub_district][city][field][disease]
								x_value = np.expand_dims(random_datapoint_for_disease[0], axis=0)
								inference = model_for_disease.predict(x_value, verbose=0)
								inference = inference[0][0]

								MSE = np.square(np.subtract([inference],[random_datapoint_for_disease[1]])).mean()
								MSE = MSE if inference < 0.0 else MSE * -1.0
								if(MSE < min_error):
									min_error = MSE
									selected_model = model_for_disease
									selected_model_strength = model_for_disease_strength
							self.CITY_MODELS[state][district][sub_district][city][disease] = [selected_model_strength, selected_model]
							count_city = count_city + 1

							if(disease not in captured_diseases_city):
								captured_diseases_city[disease] = [city]
							else:
								captured_diseases_city[disease].append(city) 

					for disease in captured_diseases_city:
						city_indices = captured_diseases_city[disease]
						min_error = 99999999999999999.0
						selected_model = None
						selected_model_strength = 0.0
						random_value = random.randint(0, 4)
						random_datapoint_for_disease = self.DISEASE_SPECIFIC_TESTING_DATA[disease][random_value]
						city = None
						for city in city_indices:
							model_for_city = self.CITY_MODELS[state][district][sub_district][city][disease][1]
							model_for_city_strength = self.CITY_MODELS[state][district][sub_district][city][disease][0]

							inference = model_for_city.predict(np.expand_dims(random_datapoint_for_disease[0], axis=0), verbose=0)[0][0]

							MSE = np.square(np.subtract([inference],[random_datapoint_for_disease[1]])).mean()
							MSE = MSE if inference < 0.0 else MSE * -1.0
							if(MSE < min_error):
								min_error = MSE
								selected_model = model_for_city
								selected_model_strength = model_for_city_strength

						self.SD_MODELS[state][district][sub_district][disease] = [selected_model_strength, selected_model]
						count_sd = count_sd + 1

						if(disease not in captured_diseases_sub_district):
							captured_diseases_sub_district[disease] = [sub_district]
						else:
							captured_diseases_sub_district[disease].append(sub_district) 

				for disease in captured_diseases_sub_district:
					sd_indices = captured_diseases_sub_district[disease]
					min_error = 99999999999999999.0
					selected_model = None
					selected_model_strength = 0.0
					random_value = random.randint(0, 4)
					random_datapoint_for_disease = self.DISEASE_SPECIFIC_TESTING_DATA[disease][random_value]
					for sd in sd_indices:
						model_for_sd = self.SD_MODELS[state][district][sd][disease][1]
						model_for_sd_strength = self.SD_MODELS[state][district][sd][disease][0]

						inference = model_for_sd.predict(np.expand_dims(random_datapoint_for_disease[0], axis=0), verbose=0)[0][0]

						MSE = np.square(np.subtract([inference],[random_datapoint_for_disease[1]])).mean()
						MSE = MSE if inference < 0.0 else MSE * -1.0
						if(MSE < min_error):
							min_error = MSE
							selected_model = model_for_sd
							selected_model_strength = model_for_sd_strength

					self.D_MODELS[state][district][disease] = [selected_model_strength, selected_model]
					count_d = count_d + 1

					if(disease not in captured_diseases_district):
						captured_diseases_district[disease] = [district]
					else:
						captured_diseases_district[disease].append(district) 

			for disease in captured_diseases_district:
				d_indices = captured_diseases_district[disease]
				min_error = 99999999999999999.0
				selected_model = None
				selected_model_strength = 0.0
				random_value = random.randint(0, 4)
				random_datapoint_for_disease = self.DISEASE_SPECIFIC_TESTING_DATA[disease][random_value]
				for d in d_indices:
					model_for_d = self.D_MODELS[state][d][disease][1]
					model_for_d_strength = self.D_MODELS[state][d][disease][0]

					inference = model_for_d.predict(np.expand_dims(random_datapoint_for_disease[0], axis=0), verbose=0)[0][0]

					MSE = np.square(np.subtract([inference],[random_datapoint_for_disease[1]])).mean()
					MSE = MSE if inference < 0.0 else MSE * -1.0
					if(MSE < min_error):
						min_error = MSE
						selected_model = model_for_d
						selected_model_strength = model_for_d_strength

				self.STATE_MODELS[state][disease] = [selected_model_strength, selected_model]
				count_state = count_state + 1

		count_b = count_city + count_sd + count_d + count_state
		N = 4
		K = self.number_of_caches;

		weight = []
		weight.append(N * 10000000.0)
		#localization_weightage = 0.1
		#communication_overhead_weightage = 2.0
		#avoid_concurrent_tiers_weightage = 5.0
		for i in range(1, N):
		  weight.append(self.localization_weightage*i)

		minWeights = []
		minWeight = 0 
		minPlacement = []

		finalmin, finalplacement = place(K, N, 999999999999999.0, [], 0.0, [], 0.0, self.communication_overhead_weightage, self.avoid_concurrent_tiers_weightage, weight)
		print(finalmin, finalplacement)
		tier_map = {"state": 4, "district": 3, "sub_district": 2, "city": 1, "field": 0}
		self.tiers_cached = []
		count = 0;

		if(1 not in finalplacement):
			self.CITY_MODELS = None;
		else :
			self.tiers_cached.append("city")
			count = count + count_city

		if(2 not in finalplacement):
			self.SD_MODELS = None
		else:
			self.tiers_cached.append("sub_district")
			count = count + count_sd

		if(3 not in finalplacement):
			self.D_MODELS = None
		else:
			self.tiers_cached.append("district")
			count = count + count_d

		if(4 not in finalplacement):
			self.STATE_MODELS = None
		else:
			self.tiers_cached.append("state")
			count = count + count_state

		end_time = datetime.now()
		print("Initialization time : ", str((end_time - start_time).total_seconds()))

		print("Number of models in b: ", str(count_b))
		print("Number of models in c: ", str(count))
		print(count_city)
		print(count_sd)
		print(count_d)
		print(count_state)


	def infer(self, test_data, tier_of_choice):
		state_names = next(os.walk('.'))[1]
		INFERENCE_RESULT = {}
		if(tier_of_choice in self.tiers_cached):
			if(tier_of_choice == 'city'):
				for state in state_names:
					if(state == '.git' or state == '__pycache__'):
						continue;
					INFERENCE_RESULT[state] = {}
					district_names = next(os.walk('./' + state))[1]
					for district in district_names:
						INFERENCE_RESULT[state][district] = {}
						sub_district_names = next(os.walk('./' + state + "/" + district))[1]
						for sub_district in sub_district_names:
							INFERENCE_RESULT[state][district][sub_district] = {}
							node_names = next(os.walk('./' + state + "/" + district + "/" + sub_district))[1]
							for city in node_names:
								INFERENCE_RESULT[state][district][sub_district][city]= {}
								for disease in self.CITY_MODELS[state][district][sub_district][city]:
									INFERENCE_RESULT[state][district][sub_district][city][disease] = [self.CITY_MODELS[state][district][sub_district][city][disease][0], self.CITY_MODELS[state][district][sub_district][city][disease][1].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]]

				return INFERENCE_RESULT
			elif(tier_of_choice == 'sub_district'):
				for state in state_names:
					if(state == '.git' or state == '__pycache__'):
						continue;
					INFERENCE_RESULT[state] = {}
					district_names = next(os.walk('./' + state))[1]
					for district in district_names:
						INFERENCE_RESULT[state][district] = {}
						sub_district_names = next(os.walk('./' + state + "/" + district))[1]
						for sub_district in sub_district_names:
							INFERENCE_RESULT[state][district][sub_district] = {}

							for disease in self.SD_MODELS[state][district][sub_district]:
								INFERENCE_RESULT[state][district][sub_district][disease] = [self.SD_MODELS[state][district][sub_district][disease][0], self.SD_MODELS[state][district][sub_district][disease][1].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]]

				return INFERENCE_RESULT

			elif(tier_of_choice == 'district'):
				for state in state_names:
					if(state == '.git' or state == '__pycache__'):
						continue;
					INFERENCE_RESULT[state] = {}
					district_names = next(os.walk('./' + state))[1]
					for district in district_names:
						INFERENCE_RESULT[state][district] = {}

						for disease in self.D_MODELS[state][district]:
							INFERENCE_RESULT[state][district][disease] = [self.D_MODELS[state][district][disease][0], self.D_MODELS[state][district][disease][1].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]]

				return INFERENCE_RESULT

			elif(tier_of_choice == 'state'):
				for state in state_names:
					if(state == '.git' or state == '__pycache__'):
						continue;
					INFERENCE_RESULT[state] = {}
					district_names = next(os.walk('./' + state))[1]
				
					for disease in self.STATE_MODELS[state]:
						INFERENCE_RESULT[state][disease] = [self.STATE_MODELS[state][disease][0], self.STATE_MODELS[state][disease][1].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]]

				return INFERENCE_RESULT
		else:
			if(tier_of_choice == "state"):
				count = 0
				if("district" in self.tiers_cached):
					STATE_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						if(state == '.git' or state == '__pycache__'):
							continue;
						STATE_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						treter = []
						captured_diseases = {}
						for district in district_names:
							for disease in self.D_MODELS[state][district]:
								captured_diseases[disease] = [self.D_MODELS[state][district][disease][0], self.D_MODELS[state][district][disease][1].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]]

							count = count + len(captured_diseases)
							treter.append(captured_diseases)
								
						
						STATE_INFER[state] = compute_weighted_inference(treter)

					print("Count123: ", str(count))
					return STATE_INFER;

				elif("sub_district" in self.tiers_cached):

					D_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						if(state == '.git' or state == '__pycache__'):
							continue;
						D_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						for district in district_names:
							D_INFER[state][district] = {}
							sub_district_names = next(os.walk('./' + state + "/" + district))[1]
							treter = []
							
							for sub_district in sub_district_names:
								captured_diseases = {}
								for disease in self.SD_MODELS[state][district][sub_district]:
									captured_diseases[disease] = [self.SD_MODELS[state][district][sub_district][disease][0], self.SD_MODELS[state][district][sub_district][disease][1].predict(np.expand_dims(test_data, axis=0), verbose = 0)[0][0]]
								
								count = count + len(captured_diseases)
								treter.append(captured_diseases)

							D_INFER[state][district] = compute_weighted_inference(treter)

					STATE_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						if(state == '.git' or state == '__pycache__'):
							continue;
						STATE_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						treter = []
						for district in district_names:
							count = count + len(D_INFER[state][district])
							treter.append(D_INFER[state][district])
								
						STATE_INFER[state] = compute_weighted_inference(treter)

					print("Count123: ", str(count))
					return STATE_INFER;
				elif("city" in self.tiers_cached):

					SD_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						if(state == '.git' or state == '__pycache__'):
							continue;
						SD_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						for district in district_names:
							SD_INFER[state][district] = {}
							sub_district_names = next(os.walk('./' + state + "/" + district))[1]
							for sub_district in sub_district_names:
								SD_INFER[state][district][sub_district] = {}
								node_names = next(os.walk('./' + state + "/" + district + "/" + sub_district))[1]
								treter = []
								for city in node_names:
									captured_diseases = {}
									for disease in self.CITY_MODELS[state][district][sub_district][city]:
										captured_diseases[disease] = [self.CITY_MODELS[state][district][sub_district][city][disease][0], self.CITY_MODELS[state][district][sub_district][city][disease][1].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]]
									count = count + len(captured_diseases)
									treter.append(captured_diseases)

								SD_INFER[state][district][sub_district] = compute_weighted_inference(treter)


					D_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						if(state == '.git' or state == '__pycache__'):
							continue;
						D_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						for district in district_names:
							D_INFER[state][district] = {}
							sub_district_names = next(os.walk('./' + state + "/" + district))[1]
							treter = []
							for sub_district in sub_district_names:
								count = count + len(SD_INFER[state][district][sub_district])
								treter.append(SD_INFER[state][district][sub_district])

							D_INFER[state][district] = compute_weighted_inference(treter)


					STATE_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						if(state == '.git' or state == '__pycache__'):
							continue;
						STATE_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						treter = []
						for district in district_names:
							count = count + len(D_INFER[state][district])
							treter.append(D_INFER[state][district])
								
						STATE_INFER[state] = compute_weighted_inference(treter)

					print("Count123: ", str(count))
					return STATE_INFER;
				else:
					return self.framework_a.infer(test_data, "state")

			elif(tier_of_choice == "district"):
				count = 0
				if("sub_district" in self.tiers_cached):

					D_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						if(state == '.git' or state == '__pycache__'):
							continue;
						D_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						for district in district_names:
							D_INFER[state][district] = {}
							sub_district_names = next(os.walk('./' + state + "/" + district))[1]
							treter = []
							
							for sub_district in sub_district_names:
								captured_diseases = {}
								for disease in self.SD_MODELS[state][district][sub_district]:
									captured_diseases[disease] = [self.SD_MODELS[state][district][sub_district][disease][0], self.SD_MODELS[state][district][sub_district][disease][1].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]]
								count = count + len(captured_diseases)
								treter.append(captured_diseases)

							D_INFER[state][district] = compute_weighted_inference(treter)
					print("Count123: ", str(count))
					return D_INFER;

				elif("city" in self.tiers_cached):
					SD_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						if(state == '.git' or state == '__pycache__'):
							continue;
						SD_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						for district in district_names:
							SD_INFER[state][district] = {}
							sub_district_names = next(os.walk('./' + state + "/" + district))[1]
							for sub_district in sub_district_names:
								SD_INFER[state][district][sub_district] = {}
								node_names = next(os.walk('./' + state + "/" + district + "/" + sub_district))[1]
								treter = []
								for city in node_names:
									captured_diseases = {}
									for disease in self.CITY_MODELS[state][district][sub_district][city]:
										captured_diseases[disease] = [self.CITY_MODELS[state][district][sub_district][city][disease][0], self.CITY_MODELS[state][district][sub_district][city][disease][1].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]]
									count = count + len(captured_diseases)
									treter.append(captured_diseases)

								SD_INFER[state][district][sub_district] = compute_weighted_inference(treter)

					D_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						if(state == '.git' or state == '__pycache__'):
							continue;
						D_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						for district in district_names:
							D_INFER[state][district] = {}
							sub_district_names = next(os.walk('./' + state + "/" + district))[1]
							treter = []
							for sub_district in sub_district_names:
								count = count + len(SD_INFER[state][district][sub_district])
								treter.append(SD_INFER[state][district][sub_district])

							D_INFER[state][district] = compute_weighted_inference(treter)
					print("Count123: ", str(count))
					return D_INFER;
				else:
					return self.framework_a.infer(test_data, "district")

			elif(tier_of_choice == "sub_district"):
				count = 0
				if("city" in self.tiers_cached):
					SD_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						if(state == '.git' or state == '__pycache__'):
							continue;
						SD_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						for district in district_names:
							SD_INFER[state][district] = {}
							sub_district_names = next(os.walk('./' + state + "/" + district))[1]
							for sub_district in sub_district_names:
								SD_INFER[state][district][sub_district] = {}
								node_names = next(os.walk('./' + state + "/" + district + "/" + sub_district))[1]
								treter = []
								for city in node_names:
									captured_diseases = {}
									for disease in self.CITY_MODELS[state][district][sub_district][city]:
										s = self.CITY_MODELS[state][district][sub_district][city][disease][0]
										s1 = self.CITY_MODELS[state][district][sub_district][city][disease][1]
										captured_diseases[disease] = [s, s1.predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]]
									count = count + len(captured_diseases)
									treter.append(captured_diseases)

								SD_INFER[state][district][sub_district] = compute_weighted_inference(treter)
					print("Count123: ", str(count))
					return SD_INFER;
				else:
					return self.framework_a.infer(test_data, "sub_district")
			elif(tier_of_choice == "city"):
				return self.framework_a.infer(test_data, "city")
			else:
				return self.framework_a.infer(test_data, "field")


