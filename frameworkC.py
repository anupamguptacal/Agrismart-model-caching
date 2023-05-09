from utils import *
from frameworkA import *

class frameworkC:

	def __init__(self,
				 number_of_caches,
				 FILE_LOCATION_DISEASE_MAPPING,
				 DISEASE_SPECIFIC_TESTING_DATA,
				 FIELD_TO_MODEL_MAPPING, 
				 communication_overhead_weightage,
				 avoid_concurrent_tiers_weightage,
				 localization_weightage):
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
		self.framework_a = frameworkA(FIELD_TO_MODEL_MAPPING)
		print("Computing Cache across tiers");
		state_names = next(os.walk('.'))[1]
		for state in state_names:
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
							index = None
							for index in field_indices:
								field = fields[index]
								models_of_field = self.FIELD_TO_MODEL_MAPPING[state][district][sub_district][city][field]

								model_for_disease = models_of_field[disease]
								x_value = np.expand_dims(random_datapoint_for_disease[0], axis=0)
								inference = model_for_disease.predict(x_value)
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

							inference = model_for_city.predict(np.expand_dims(random_datapoint_for_disease[0], axis=0))[0][0]

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

						inference = model_for_sd.predict(np.expand_dims(random_datapoint_for_disease[0], axis=0))[0][0]

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

					inference = model_for_d.predict(np.expand_dims(random_datapoint_for_disease[0], axis=0))[0][0]

					MSE = np.square(np.subtract([inference],[random_datapoint_for_disease[1]])).mean()
					MSE = MSE if inference < 0.0 else MSE * -1.0
					if(MSE < min_error):
						min_error = MSE
						selected_model = model_for_d

				self.STATE_MODELS[state][disease] = selected_model


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
		#finalplacement = [1,3]
		tier_map = {"state": 4, "district": 3, "sub_district": 2, "city": 1, "field": 0}
		self.tiers_cached = []

		if(1 not in finalplacement):
			self.CITY_MODELS = None;
		else :
			self.tiers_cached.append("city")

		if(2 not in finalplacement):
			self.SD_MODELS = None
		else:
			self.tiers_cached.append("sub_district")

		if(3 not in finalplacement):
			self.D_MODELS = None
		else:
			self.tiers_cached.append("district")

		if(4 not in finalplacement):
			self.STATE_MODELS = None
		else:
			self.tiers_cached.append("state")


	def infer(self, test_data, tier_of_choice):
		if(tier_of_choice in self.tiers_cached):
			if(tier_of_choice == 'city'):
				INFERENCE_RESULT = {}
				for state in state_names:
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
									INFERENCE_RESULT[state][district][sub_district][city][disease] = self.CITY_MODELS[state][district][sub_district][city][disease].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]

				return INFERENCE_RESULT
			elif(tier_of_choice == 'sub_district'):
				for state in state_names:
					INFERENCE_RESULT[state] = {}
					district_names = next(os.walk('./' + state))[1]
					for district in district_names:
						INFERENCE_RESULT[state][district] = {}
						sub_district_names = next(os.walk('./' + state + "/" + district))[1]
						for sub_district in sub_district_names:
							INFERENCE_RESULT[state][district][sub_district] = {}

							for disease in self.SD_MODELS[state][district][sub_district]:
								INFERENCE_RESULT[state][district][sub_district][disease] = self.SD_MODELS[state][district][sub_district][disease].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]

				return INFERENCE_RESULT

			elif(tier_of_choice == 'district'):
				for state in state_names:
					INFERENCE_RESULT[state] = {}
					district_names = next(os.walk('./' + state))[1]
					for district in district_names:
						INFERENCE_RESULT[state][district] = {}

						for disease in self.D_MODELS[state][district]:
							INFERENCE_RESULT[state][district][disease] = self.D_MODELS[state][district][disease].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]

				return INFERENCE_RESULT

			elif(tier_of_choice == 'state'):
				for state in state_names:
					INFERENCE_RESULT[state] = {}
					district_names = next(os.walk('./' + state))[1]
				
					for disease in self.STATE_MODELS[state]:
						INFERENCE_RESULT[state][disease] = self.STATE_MODELS[state][disease].predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]

				return INFERENCE_RESULT
		else:
			if(tier_of_choice == "state"):
				if("district" in self.tiers_cached):


					STATE_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						STATE_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						treter = []
						captured_diseases = {}
						for district in district_names:
							for disease in self.D_MODELS[state][district]:
								captured_diseases[disease] = [1, self.D_MODELS[state][district][disease].predict(np.expand_dims(test_data, axis=0))[0][0]]

							treter.append(captured_diseases)
								
						
						STATE_INFER[state] = compute_weighted_inference(treter)

					return STATE_INFER;

				elif("sub_district" in self.tiers_cached):

					D_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						D_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						for district in district_names:
							D_INFER[state][district] = {}
							sub_district_names = next(os.walk('./' + state + "/" + district))[1]
							treter = []
							
							for sub_district in sub_district_names:
								captured_diseases = {}
								for disease in self.SD_MODELS[state][district][sub_district]:
									captured_diseases[disease] = [1, self.SD_MODELS[state][district][sub_district][disease].predict(np.expand_dims(test_data, axis=0))[0][0]]
								treter.append(captured_diseases)

							D_INFER[state][district] = compute_weighted_inference(treter)

					STATE_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						STATE_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						treter = []
						for district in district_names:
							treter.append(D_INFER[state][district])
								
						STATE_INFER[state] = compute_weighted_inference(treter)

					return STATE_INFER;
				elif("city" in self.tiers_cached):

					SD_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
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
									for disease in self.CITY_MODELS[state][district][sub_district]:
										captured_diseases[disease].append([1, self.CITY_MODELS[state][district][sub_district][disease].predict(np.expand_dims(test_data, axis=0))[0][0]])
									treter.append(captured_diseases)

							SD_INFER[state][district][sub_district] = compute_weighted_inference(treter)


					D_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						D_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						for district in district_names:
							D_INFER[state][district] = {}
							sub_district_names = next(os.walk('./' + state + "/" + district))[1]
							treter = []
							for sub_district in sub_district_names:
								treter.append(SD_INFER[state][district][sub_district])

							D_INFER[state][district] = compute_weighted_inference(treter)


					STATE_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						STATE_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						treter = []
						for district in district_names:
							treter.append(D_INFER[state][district])
								
						STATE_INFER[state] = compute_weighted_inference(treter)

					return STATE_INFER;
				else:
					return self.framework_a.infer(test_data, "state")

			elif(tier_of_choice == "district"):
				if("sub_district" in self.tiers_cached):

					D_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
						D_INFER[state] = {}
						district_names = next(os.walk('./' + state))[1]
						for district in district_names:
							D_INFER[state][district] = {}
							sub_district_names = next(os.walk('./' + state + "/" + district))[1]
							treter = []
							
							for sub_district in sub_district_names:
								captured_diseases = {}
								for disease in self.SD_MODELS[state][district][sub_district]:
									captured_diseases[disease] = [1, self.SD_MODELS[state][district][sub_district][disease].predict(np.expand_dims(test_data, axis=0))[0][0]]
								treter.append(captured_diseases)

							D_INFER[state][district] = compute_weighted_inference(treter)
					return D_INFER;

				elif("city" in self.tiers_cached):
					SD_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
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
									for disease in self.CITY_MODELS[state][district][sub_district]:
										captured_diseases[disease].append([1, self.CITY_MODELS[state][district][sub_district][disease].predict(np.expand_dims(test_data, axis=0))[0][0]])
									treter.append(captured_diseases)

							SD_INFER[state][district][sub_district] = compute_weighted_inference(treter)


						D_INFER = {}
						state_names = next(os.walk('.'))[1]
						for state in state_names:
							D_INFER[state] = {}
							district_names = next(os.walk('./' + state))[1]
							for district in district_names:
								D_INFER[state][district] = {}
								sub_district_names = next(os.walk('./' + state + "/" + district))[1]
								treter = []
								for sub_district in sub_district_names:
									treter.append(SD_INFER[state][district][sub_district])

								D_INFER[state][district] = compute_weighted_inference(treter)
					return D_INFER;
				else:
					return self.framework_a.infer(test_data, "district")

			elif(tier_of_choice == "sub_district"):
				if("city" in self.tiers_cached):
					SD_INFER = {}
					state_names = next(os.walk('.'))[1]
					for state in state_names:
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
									for disease in self.CITY_MODELS[state][district][sub_district]:
										captured_diseases[disease].append([1, self.CITY_MODELS[state][district][sub_district][disease].predict(np.expand_dims(test_data, axis=0))[0][0]])
									treter.append(captured_diseases)

							SD_INFER[state][district][sub_district] = compute_weighted_inference(treter)
					return SD_INFER;
				else:
					return self.framework_a.infer(test_data, "sub_district")
			elif(tier_of_choice == "city"):
				return self.framework_a.infer(test_data, "city")
			else:
				return self.framework_a.infer(test_data, "field")


