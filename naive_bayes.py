import math
import pandas
import sys
import functools

probability_tuples = []
starting_probabilities = []
feature_probabilities = [] # probability of a feature being true (will be multiplied to yield the denominator later)
SMOOTHING = 1.1

# calculate probability of every individual feature given every class_type 1-7
def calculate_probabilities(train):
	features = train.columns.tolist()
	for class_type in range(1, 8):
		starting_probabilities.append(len(train[(train['class_type'] == class_type)]) / len(train))
		for feature in features[1:17]:
			if feature == 'legs':
				for count in range(9):
					if class_type == 1:
						feature_probabilities.append( (str(count) + ' legs', len(train[(train[feature] == count)]) / len(train)) )

					probability = (len(train[(train['class_type'] == class_type) & (train[feature] == count)]) + SMOOTHING) / (len(train[(train['class_type'] == class_type)]) + SMOOTHING)
					new_tuple = (str(count) + ' legs', class_type, probability)
					probability_tuples.append(new_tuple)
			else:
				if class_type == 1:
					feature_probabilities.append((feature, len(train[(train[feature] == 1)]) / len(train)))

				# P(feature | class_type) = number of animals with class type having the feature / total number of animals having that class_type
				probability = (len(train[(train['class_type'] == class_type) & (train[feature] == 1)]) + SMOOTHING) / (len(train[(train['class_type'] == class_type)]) + SMOOTHING)
				# add this probability to a tuple specifying the feature and class_type, then add that tuple to the main list
				new_tuple = (feature, class_type, probability)
				probability_tuples.append(new_tuple)

# use previously calculated probabilities (multiply them to obtain and compare likelihoods for each class_type)
def assign_classes(test):
	correct = 0
	total = 0
	for index, row in test.iterrows():
		denominator = 1

		class_probabilities = starting_probabilities.copy() # a list of 7 probabilities, the class one corresponds to is index + 1
		for feature in test.columns.tolist()[1:17]:
			value = row[feature]

			for match in feature_probabilities:
				if value == 1 and match[0] == feature:
					denominator *= match[1]
					break
				elif feature == 'legs' and str(value) + ' legs' == match[0]:
					denominator *= match[1]
					break

			
			if feature == 'legs':
				for probability_tuple in probability_tuples:
					if probability_tuple[0][0] == str(value):
						class_probabilities[probability_tuple[1] - 1] *= probability_tuple[2]
			else:
				for probability_tuple in probability_tuples:
					if value == 1 and probability_tuple[0] == feature:
						class_probabilities[probability_tuple[1] - 1] *= probability_tuple[2]
		
		for probability in class_probabilities:
			probability = (probability + 0.001) / (denominator + 0.001)

		if row['class_type'] == class_probabilities.index(max(class_probabilities)) + 1:
			result = 'CORRECT'
			correct += 1
		else:
			result = 'wrong'
		total += 1
		test.loc[index, 'predicted'] = str(class_probabilities.index(max(class_probabilities)) + 1)
		test.loc[index, 'probability'] = max(class_probabilities)
		test.loc[index, 'correct?'] = result
	print(test)
	print('Accuracy: ' + str(correct/total * 100) + '%')

def main():
	data = pandas.read_csv('zoo.csv')
	train = data.sample(frac = 0.7)
	test = data.drop(train.index)
	
	calculate_probabilities(train)
	assign_classes(test)

if __name__ == '__main__':
	main()