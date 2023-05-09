from utils import *
class frameworkA:	

	def __init__(self,
	 			FIELD_TO_MODEL_MAPPING):
		self.FIELD_TO_MODEL_MAPPING = FIELD_TO_MODEL_MAPPING
		print_separation_line()
		self.tiers = ['field', 'city', 'sub_district', 'district', 'state']

	def infer(self, test_data, tier_of_choice):
		found = False
		for i in range(len(self.tiers)):
			if(tier_of_choice == self.tiers[i]) :
				found = True;
				break;
		if(not found) :
			return [-1]

		level = 0
		INFERENCE = {}
		disease_specific_inferences = {}
		state_names = next(os.walk('.'))[1]
		for state in state_names:
			INFERENCE[state] = {}
			district_names = next(os.walk('./' + state))[1]
			for district in district_names:
				INFERENCE[state][district] = {}
				sub_district_names = next(os.walk('./' + state + "/" + district))[1]
				for sub_district in sub_district_names:
					INFERENCE[state][district][sub_district] = {}
					node_names = next(os.walk('./' + state + "/" + district + "/" + sub_district))[1]
					for city in node_names:
						INFERENCE[state][district][sub_district][city] = {}
						fields = next(os.walk('./' + state + '/' + district + "/" + sub_district + "/" + city))[1]
						for field in fields:
							models_in_field = self.FIELD_TO_MODEL_MAPPING[state][district][sub_district][city][field]
							disease_dict = {}
							for disease in models_in_field:
								a = []
								model_for_disease = models_in_field[disease]
								inference_value = model_for_disease.predict(np.expand_dims(test_data, axis=0), verbose=0)[0][0]

								y_train_location = construct_train_y_file_location(state, district, sub_district, city, field)
								strength = 0.0
								with open(y_train_location,"r") as f:
									strength = len(f.readlines())
								a.append(strength)
								a.append(inference_value)

								disease_dict[disease] = a;

							INFERENCE[state][district][sub_district][city][field] = disease_dict

		if(tier_of_choice == 'field'):
			return INFERENCE

		CITY_INFER = {}
		state_names = next(os.walk('.'))[1]
		for state in state_names:
			CITY_INFER[state] = {}
			district_names = next(os.walk('./' + state))[1]
			for district in district_names:
				CITY_INFER[state][district] = {}
				sub_district_names = next(os.walk('./' + state + "/" + district))[1]
				for sub_district in sub_district_names:
					CITY_INFER[state][district][sub_district] = {}
					node_names = next(os.walk('./' + state + "/" + district + "/" + sub_district))[1]
					for city in node_names:
						fields = next(os.walk('./' + state + '/' + district + "/" + sub_district + "/" + city))[1]
						treter = []
						for field in fields:
							treter.append(INFERENCE[state][district][sub_district][city][field])

						CITY_INFER[state][district][sub_district][city] = compute_weighted_inference(treter)

		if(tier_of_choice == 'city'):
			return CITY_INFER

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
						treter.append(CITY_INFER[state][district][sub_district][city])

				#print(treter)
				SD_INFER[state][district][sub_district] = compute_weighted_inference(treter)

		if(tier_of_choice == 'sub_district'):
			return SD_INFER


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

		if(tier_of_choice == 'district'):
			return D_INFER


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