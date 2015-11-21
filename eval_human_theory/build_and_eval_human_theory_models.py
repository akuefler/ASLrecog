# Quick and dirty; moved from ipython into a standalone script. Refactor to 
# your heart's desire. In particular, we may want to separate some 
# of these functions out into distinct files that can then be included or
# not included in new scripts as we move forward, and we may want to make
# this more beautiful Python by not invoking in the dumb way below. 
# But, this is fast-to-get-results research code, not production code, so 
# no guarantees of beauty.

from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics
from sklearn import svm
from sklearn import naive_bayes
from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def import_and_split(featureName):
    # Get training data
    training_data = pd.read_csv("transposed_results.csv", header=0, \
                                index_col=0, na_values='?')

    # Figure out the rows of examples for which we actually know if they are wh questions or not
    legitData = np.isfinite(training_data[featureName])

    # Create a training data set from those rows
    the_training_data = training_data[legitData]
    the_truth = the_training_data[featureName]
    del the_training_data[featureName]
    return the_training_data, the_truth

def remove_additional_features(the_training_data, listOfFeatures):
    for l in listOfFeatures:
        if l in the_training_data:
            del the_training_data[l]
    return the_training_data
	

# Option 1: Fill NA with 0s (aka, wasn't important enough to be populated)
def replace_na_with_zero(the_training_data):
    the_training_data = the_training_data.fillna(0)
    return the_training_data
	

# Option 2 -- drop all missing columns -- nothing persists :(

# Params:
#  the_training_data: init training data
#  style: 0 indicates evaluation, other numbers indicate number of features to keep
# Returns:
#  nothing if in eval mode; revised version of the_training_data
def consider_features(the_training_data, style=0):
    # So, figure out which features to use by iteratively dropping
    # the smallest-used feature
    countsByFeature = the_training_data.count(axis='rows')
    countsByFeature.sort()
    labelsInOrderOfBigness = countsByFeature.index

    # For each label in order of bigness
    #     Remove it from the data & re-save data on top of countsByFeature
    #     Drop any rows in the data that contain any NAs
    #     Drop the NA data and see how much data we have left
    #     Store the x-y pair: numFeaturesInDataset; numTrainingExamplesLeft
    numTraining = []
    numFeature = []
    for l in labelsInOrderOfBigness:
        del the_training_data[l]

        temp = the_training_data.dropna(axis='rows')

        countsByFeature = temp.count(axis='rows')
        countsByFeature.sort()

        labelsInOrderOfBigness = countsByFeature.index

        numTraining.append(np.shape(temp)[0])
        numFeature.append(np.shape(temp)[1])

        # Stop when we've gotten the right number of features features
        if (style > 0 and np.shape(temp)[1] == style):
            return the_training_data
        elif (style == 0):
            print np.shape(temp)

    # Plot:
    # (numFeaturesLeft, numTrainingExamplesLeft)
    if style==0:
        #print numTrainingX
        #print numFeatureY
        plt.scatter(numFeature, numTraining)
        plt.show()

    # Choose a number of features that gives a 
    # reasonable number of training examples
    #
    # Data-driven best options seem to be...
    # 21 features, 97 examples
    # 17 features 1256 examples
    # 12 features, 2968 examples
    # 6 features, 8202 examples


# Given a dataset where many are NAs, drop the examples with any NAs
# to get only the 17 features that occur in 1256 training examples

def drop_nas(the_training_data, the_truth):
    # Create training data
    the_training_data = the_training_data.dropna(axis='rows')
    # Create truth
    the_truth = the_truth[the_training_data.index]

    print "Number examples, number of positive examples, proportion of 1s:"
    print len(the_truth), sum(the_truth), sum(the_truth)/len(the_truth)
    
    return the_training_data, the_truth


# Return a list of: accuracy,accuracystdev,numFeatures,numExamples,numUtterances
def eval_log_reg(the_training_data, the_truth): 
    K_FOLD = 10
    
    # Linear regression
    lr = linear_model.LogisticRegression()

    # Evaluate
    scores = cross_validation.cross_val_score(lr, the_training_data, the_truth, cv=K_FOLD)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    predicted = cross_validation.cross_val_predict(lr, the_training_data, the_truth, cv=K_FOLD)
    print "Confusion matrix:"
    print metrics.confusion_matrix(the_truth, predicted)
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        the_training_data, the_truth, test_size=1.0/K_FOLD, random_state=0)
    lr.fit(X_train, y_train)
    labels = X_train.columns
    coefficients = [(labels[i],val) for i,val in enumerate(lr.__dict__['coef_'][0])]
    coefficients.sort(key=lambda x: abs(x[1]), reverse=True)
    print "Most predictive features:"
    for i in range(0,5):
        print "    %s: %0.2f" % (coefficients[i][0], coefficients[i][1])
    
    numExamples = np.shape(X_train)[0]
    print "Training examples: %d" % numExamples
    usedUtterances = [example.split(".csv_")[0] for example in X_train.index]
    numUtterances = len(set(usedUtterances))
    print  "Training utterances: %d" % numUtterances
    
    return [scores.mean(), scores.std() * 2, len(coefficients), numExamples, numUtterances]
	
	
def getNumUtterances(theSet):
    usedUtterances = [example.split(".csv_")[0] for example in theSet.index]
    numUtterances = len(set(usedUtterances))
    return numUtterances 

# Return a list of: accuracy,accuracystdev,numFeatures,numExamples,numUtterances
def eval_correct(model,the_training_data, the_truth): 
    K_FOLD = 10
    
    # Build train & test sets
    confusion_matrix = np.zeros((2,2))
    scores = np.zeros((1,K_FOLD))
    utterances = [example.split(".csv_")[0] for example in the_training_data.index]
    lkf = cross_validation.LabelKFold(utterances, n_folds=K_FOLD)
    for i, (trainIdx, testIdx) in enumerate(lkf):
        trainX = the_training_data.iloc[trainIdx]
        trainY = the_truth.iloc[trainIdx]
        testX = the_training_data.iloc[testIdx]
        testY = the_truth.iloc[testIdx]
        
        model.fit(trainX, trainY)
        predictedY = model.predict(testX)
        scores[0,i] = model.score(testX, testY)
        confusion_matrix += metrics.confusion_matrix(testY, predictedY)
    
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    print "Confusion matrix:"
    print confusion_matrix
    
    attrib = model.__dict__
    if ("coef_" in attrib):
        featureLabels = trainX.columns
        coefficients = [(featureLabels[i],val) for i,val in enumerate(attrib['coef_'][0])]
        coefficients.sort(key=lambda x: abs(x[1]), reverse=True)
        print "Most predictive features:"
        for i in range(0,5):
            print "    %s: %0.2f" % (coefficients[i][0], coefficients[i][1])
    
    numExamplesTrain = np.shape(trainX)[0]
    numExamplesTest = np.shape(testX)[0]
    print "Training examples: %d" % numExamplesTrain
    print "Test examples: %d" % numExamplesTest
    
    print  "Training utterances: %d" % getNumUtterances(trainX)
    print  "Test utterances: %d" % getNumUtterances(testX)
    
    
    return [scores.mean(), scores.std() * 2, len(trainX.columns), \
            numExamplesTrain, numExamplesTest, \
            getNumUtterances(trainX), getNumUtterances(testX)]
			
# Return a list of lists of: accuracy,accuracystdev,numFeatures,
#    numExamplesTrain,numExamplesTest,numUtterancesTrain,numUtterancesTest
def compare_feature_sets(model, the_training_data_orig, the_truth_orig, numFeatures):
    results = []
    # Logistic regression using 0s instead of NAs
    print "== Using all features and replacing NAs in samples with 0s == "
    the_training_data = replace_na_with_zero(the_training_data_orig)
    results.append(eval_correct(model, the_training_data, the_truth_orig))
    print

    # Logistic regression on varying numbers of features
    for numFeature in numFeatures:
        print ("== Using %d features == " % numFeature)
        the_training_data = consider_features(the_training_data_orig, style=numFeature)
        the_training_data, the_truth = drop_nas(the_training_data, the_truth_orig)
        results.append(eval_correct(model, the_training_data, the_truth))
        print
        
    return results


def run_evaluation():
	goalFeature = "wh question => (whq)"
	cheatingFeatures = ["rhetorical question => (wh rhq)", "POS => (Wh-word)", \
						"Non-dominant POS => (Wh-word)", "conditional/when => (when)", \
						"POS2 => (Wh-word)"]

	the_training_data_orig, the_truth_orig = import_and_split(goalFeature)
	training_data_non_cheat = remove_additional_features(the_training_data_orig, cheatingFeatures)

	run_labels = []
	performance_results = []
	models = {"Logistic regression": linear_model.LogisticRegression(), 
			  "SVM - Linear Kernel": svm.SVC(kernel='linear'), 
			  "SVM - Polynomial Kernel": svm.SVC(kernel='poly'), 
			  "SVM - RBF Kernel": svm.SVC(kernel='rbf'),
			  "Gaussian NB": naive_bayes.GaussianNB(),
			  "Decision Tree": tree.DecisionTreeClassifier(),
			  "Perceptron": linear_model.Perceptron()
			  }

	for modelId in models:
		print "####################"
		print modelId
		print "####################"
		model = models[modelId]
		res = compare_feature_sets(model, the_training_data_orig.copy(), the_truth_orig.copy(), [16, 11, 7])
		performance_results.extend(res)
		run_labels.append(modelId)


	# accuracy,accuracystdev,numFeatures,
	# numExamplesTrain,numExamplesTest,numUtterancesTrain,numUtterancesTest
	print run_labels
	for run in performance_results:
		print "%0.2f %0.2f %d %d %d %d %d" % (run[0], run[1], run[2], run[3], run[4], run[5], run[6])
		
		

run_evaluation()