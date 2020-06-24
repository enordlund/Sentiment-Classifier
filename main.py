import codecs
import math


def writeStringToFileUTF8(string, filepath):
	file = codecs.open(filepath, "w", "utf-8")
	
	# write string
	file.write(string)
	
	file.close()


# PRE-PROCESSING

punctuationCharacters = set(("^", "~", "`", "'"))
wordSeperatorCharacters = set(("-", "_", "–", "—", " ", ".", ",", "?", "!", ":", ";", "/", "\"", "`", "<", ">", "=", "+", "|", "@", "#", "$", "%", "&", "*", "(", ")", "{", "}", "[", "]", "\\"))
vocabulary = set() #becomes sorted array with setVocabulary()

featurizedTrainingData = list()
featurizedTestData = list()

def preProcessor():
	print("Processing data...")
	global featurizedTrainingData
	global featurizedTestData
	
	trainingFileLines = getLines("trainingSet.txt")
	
	# the lines will still include tabs ("\t") and newlines ("\n"), in a list.
	
	# go through the lines, stripping punctuation and adding new words to vocabulary
	setVocabulary(trainingFileLines)
#	print(vocabulary)
	
	
	# create featurized data for training set and testing set
	featurizedTrainingData = getFeatureVectors(trainingFileLines)
	
	testFileLines = getLines("testSet.txt")
	featurizedTestData = getFeatureVectors(testFileLines)
	
#	print(featurizedTestData)
	
	# print the data to the files
	printPreProcessedData("preprocessed_train.txt", featurizedTrainingData)
	printPreProcessedData("preprocessed_test.txt", featurizedTestData)
	
	

def getLines(fromFilePath):
	trainingFile = open(fromFilePath, encoding='utf-8')
	
	lines = trainingFile.readlines() 
	
	trainingFile.close()
	
	return lines
	


def setVocabulary(fromLines):
	global vocabulary
	
	for line in fromLines:
		addLineToVocab(line)
	
	# convert the set to an alphabetically-sorted list
	vocabulary = list(vocabulary)
	vocabulary.sort()

def addLineToVocab(fromLine):
	# step through words, stripping punctuation and adding them to vocabulary if new
	word = str()
	for character in fromLine:
		if character not in punctuationCharacters:
			# character is not punctuation
			if character == "\t":
				# end of words in line (sentiment value follows)
				return
			elif character in wordSeperatorCharacters:
				# end of the current word
				if len(word) > 0:
					# word is stored
					vocabulary.add(word)
				word = str()
			else:
				# character should be added to current word
				word += character.lower()
				
	return



def getFeatureVectors(fromLines):
	vectors = list()
	
	for line in fromLines:
		vector = getFeatureVector(line)
		vectors.append(vector)
	
	return vectors


def getWords(fromLine):
	lineWords = list()
	
	word = str()
	for character in fromLine:
#		print(character)
		if character not in punctuationCharacters:
			# character is not punctuation
			if character == "\t":
				# end of words in line (sentiment value follows)
				return lineWords
			elif character in wordSeperatorCharacters:
				# end of the current word
				if len(word) > 0:
					# word is stored
#					print("appending")
					lineWords.append(word)
				word = str()
			else:
				# character should be added to current word
				word += character.lower()
	
	return lineWords

def getFeatureVector(fromLine):
	# get words from the line
	lineWords = getWords(fromLine)
	
	# assuming vocabulary is an array by now...
	# populate the vector with len(vocabulary) zeros (sentiment appended later in function)
	vector = [0 for i in range(len(vocabulary))]
#	print(fromLine)
#	print(lineWords)
	
	# set flags for words in line
	for word in lineWords:
		# if the word is in the vocabulary, set the flag
		try:
			wordIndex = vocabulary.index(word)
		except:
			continue
		vector[wordIndex] = 1
	
	# line sentiment is the third-to-last character in each line (preceeding " \n").
	sentimentIndex = len(fromLine) - 3
	sentiment = int(fromLine[sentimentIndex])
	
	# append sentiment to vector
	vector.append(sentiment)
	
	return vector


def printPreProcessedData(toFilepath, featureVectors):
	file = codecs.open(toFilepath, "w", "utf-8")
	
	# first, print vocab
	vocabListString = getListString(vocabulary, True)
	file.write(vocabListString)
	
	# print feature vectors
	for vector in featureVectors:
		vectorString = getListString(vector, False)
		file.write(vectorString)
	
	file.close()


def getListString(fromList, withClasslabel):
	listString = str()
	
	# add words to list
	for string in fromList:
		listString += str(string)
		listString += ","
	
	if withClasslabel:
		# add dummy non-word "classlabel"
		listString += "classlabel"
	else:
		# remove trailing comma
		listString = listString[:-1]
	
	listString += "\n"
	
	return listString
	











# CLASSIFIER TRAINING

# values for model
probabilityOfPositiveSentiment = 0.0
probabilityOfNegativeSentiment = 0.0

# P(F|S), with F being a feature and S being the sentiment
probabilityOfPresentFeatureGivenPositiveSentiment = list()
probabilityOfAbsentFeatureGivenPositiveSentiment = list()

probabilityOfPresentFeatureGivenNegativeSentiment = list()
probabilityOfAbsentFeatureGivenNegativeSentiment = list()

def initializeConditionalLists():
	featureCount = len(featurizedTestData[0]) - 1
	
	global probabilityOfPresentFeatureGivenPositiveSentiment
	global probabilityOfAbsentFeatureGivenPositiveSentiment
	
	global probabilityOfPresentFeatureGivenNegativeSentiment
	global probabilityOfAbsentFeatureGivenNegativeSentiment
	
	probabilityOfPresentFeatureGivenPositiveSentiment = [0.0] * featureCount
	probabilityOfAbsentFeatureGivenPositiveSentiment = [0.0] * featureCount
	
	probabilityOfPresentFeatureGivenNegativeSentiment = [0.0] * featureCount
	probabilityOfAbsentFeatureGivenNegativeSentiment = [0.0] * featureCount


def calculateSentimentProbabilities():
	global featurizedTrainingData
	
	dataCount = len(featurizedTrainingData)
	
	positiveSentimentCount = 0
	
	
	for vector in featurizedTrainingData:
		# adding 1 to positiveSentimentCount when it's positive, or 0 when it's negative.
		positiveSentimentCount += vector[-1]
	
	# calculate the probabilities
	global probabilityOfPositiveSentiment
	global probabilityOfNegativeSentiment
	
	probabilityOfPositiveSentiment = float(positiveSentimentCount) / float(dataCount)
	probabilityOfNegativeSentiment = 1.0 - probabilityOfPositiveSentiment



def calculateConditionalProbabilities():
	global featurizedTrainingData
	
	global probabilityOfPresentFeatureGivenPositiveSentiment
	global probabilityOfAbsentFeatureGivenPositiveSentiment
	
	global probabilityOfPresentFeatureGivenNegativeSentiment
	global probabilityOfAbsentFeatureGivenNegativeSentiment
	
	initializeConditionalLists()
	
	dataCount = len(featurizedTrainingData)
	positiveSentimentCount = 0
	
	for vector in featurizedTrainingData:
		# remove the label for simplified feature loop
		features = vector[:-1]
		label = vector[-1]
		positiveSentimentCount += label# count positive sentiment labels
			
		
		for index, feature in enumerate(features):
			if feature == 1:
				# feature is present
				if label == 1:
					# sentiment is positive
					# add 1 to index of feature in
					# probabilityOfPresentFeatureGivenPositiveSentiment
					probabilityOfPresentFeatureGivenPositiveSentiment[index] += 1
				else:
					# sentiment is negative
					# add 1 to index of feature in
					# probabilityOfPresentFeatureGivenNegativeSentiment
					probabilityOfPresentFeatureGivenNegativeSentiment[index] += 1
		
	# now the present feature lists are populated with sums. Calculate probabilities with Dirichlet priors.
	negativeSentimentCount = dataCount - positiveSentimentCount
	
	for index, value in enumerate(probabilityOfPresentFeatureGivenPositiveSentiment):
		presentProbability = float(value + 1.0) / float(positiveSentimentCount + 2.0)
		probabilityOfPresentFeatureGivenPositiveSentiment[index] = presentProbability
		
		# set the absent probability to the compliment of present
		probabilityOfAbsentFeatureGivenPositiveSentiment[index] = 1.0 - presentProbability
		
	for index, value in enumerate(probabilityOfPresentFeatureGivenNegativeSentiment):
		presentProbability = float(value + 1.0) / float(negativeSentimentCount + 2.0)
		probabilityOfPresentFeatureGivenNegativeSentiment[index] = presentProbability
		
		# set absent probability to the compliment of present
		probabilityOfAbsentFeatureGivenNegativeSentiment[index] = 1.0 - presentProbability
	
	# at this point, all four conditional probability lists are populated.



def trainModel():
	print("Training...")
	calculateSentimentProbabilities()
	calculateConditionalProbabilities()
	
	print("Done training")
	
#	print("Positive Probability: " + str(probabilityOfPositiveSentiment))
#	print("Negative Probability: " + str(probabilityOfNegativeSentiment))
#	
#	print("Probability lists:")
#	print(probabilityOfPresentFeatureGivenPositiveSentiment)
#	print(probabilityOfAbsentFeatureGivenPositiveSentiment)
#	print(probabilityOfPresentFeatureGivenNegativeSentiment)
#	print(probabilityOfAbsentFeatureGivenNegativeSentiment)



# CLASSIFIER TESTING

# written assuming forVector is not empty.
def getVectorLabel(forVector):
	# return last element, which represents positive or negative with 1 or 0.
	return forVector[-1]



# written assuming forVector is not empty.
def getClassifierResult(forVector):
	# analyze with trained model
	
	global probabilityOfPresentFeatureGivenPositiveSentiment
	global probabilityOfAbsentFeatureGivenPositiveSentiment
	
	global probabilityOfPresentFeatureGivenNegativeSentiment
	global probabilityOfAbsentFeatureGivenNegativeSentiment
	
	global probabilityOfPositiveSentiment
	global probabilityOfNegativeSentiment
	
	# remove label from vector
	features = forVector[:-1]
	
	positiveProductElements = list()
	negativeProductElements = list()
	
	for index, feature in enumerate(features):
		if feature == 1:
			# feature is present
			positiveProductElement = probabilityOfPresentFeatureGivenPositiveSentiment[index]
			negativeProductElement = probabilityOfPresentFeatureGivenNegativeSentiment[index]
			
			positiveProductElements.append(positiveProductElement)
			negativeProductElements.append(negativeProductElement)
		else:
			# feature is absent
			positiveProductElement = probabilityOfAbsentFeatureGivenPositiveSentiment[index]
			negativeProductElement = probabilityOfAbsentFeatureGivenNegativeSentiment[index]
			
			positiveProductElements.append(positiveProductElement)
			negativeProductElements.append(negativeProductElement)
	
	# calculate probabilities
	positiveProbability = math.log(probabilityOfPositiveSentiment)
	
	for element in positiveProductElements:
		positiveProbability = positiveProbability + math.log(element)
	
	negativeProbability = math.log(probabilityOfNegativeSentiment)
	
	for element in negativeProductElements:
		negativeProbability = negativeProbability + math.log(element)
	
	# compare probabilities
	if positiveProbability > negativeProbability:
		return 1
	else:
		# default to negative if probabilities are equal
		return 0
	
	return 1


def testClassifierResult(forVector):
	if forVector:
		# vector is not empty
		vectorLabel = getVectorLabel(forVector)
		classifierResult = getClassifierResult(forVector)
		
		if vectorLabel == classifierResult:
			return True
		else:
			return False

def testClassifier(onData):
	print("Testing classifier...")
	
	accurateClassificationsCount = 0
	totalClassificationsCount = 0
	
	for vector in onData:
		testResult = testClassifierResult(vector)
		# no exception was thrown, so handle testResult
		if testResult == True:
			# classifier result was accurate
			accurateClassificationsCount += 1
			totalClassificationsCount += 1
		elif testResult == False:
			# classifier was not accurate
			totalClassificationsCount += 1
		else:
			print("test failed")
			
	
	# end of test
	# calculate accuracy
	accuracy = (float(accurateClassificationsCount) / float(totalClassificationsCount)) * 100
	
	# write to the output file
	accuracyString = "Accuracy = " + str(accuracy) + "%"
	
	print(accuracyString)
	
	return accuracyString










# EXECUTION
preProcessor()

trainModel()

trainingDataAccuracyString = testClassifier(featurizedTrainingData)
testDataAccuracyString = testClassifier(featurizedTestData)

writeStringToFileUTF8("Training Data " + trainingDataAccuracyString + "\n" + "Test Data " + testDataAccuracyString, "results.txt")
print("Written to results.txt")