from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics
from sklearn import svm
from sklearn import naive_bayes
from sklearn import tree
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

# SAME AS EVAL_HUMAN_THEORY
def getNumUtterances(theSet):
    usedUtterances = [example.split(".csv_")[0] for example in theSet.index]
    numUtterances = len(set(usedUtterances))
    return numUtterances 

# SAME AS EVAL_HUMAN_THEORY
# Return a list of: accuracy,accuracystdev,numFeatures,numExamples,numUtterances
def eval_correct(model, the_training_data, the_truth): 
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
    
    print
    
    return [scores.mean(), scores.std() * 2, len(trainX.columns), 
            numExamplesTrain, numExamplesTest, 
            getNumUtterances(trainX), getNumUtterances(testX)]
			
			
####################################
# New code
####################################

def import_and_split_intraface(featureName):
    # Get training data
    training = pd.read_csv("rawIntraface.csv", header=0, \
                                index_col=0, na_values='?')
    # Get truth column
    truth = pd.read_csv("transposed_results.csv", header=0, \
                                index_col=0, na_values='?')
    truth = truth[featureName]
    
    # Remove rows with no truth by way of an SQL-like merge on row names
    training = pd.DataFrame(training)
    truth = pd.DataFrame(truth)
    both = pd.concat([training, truth], axis=1, join='outer')
    legitData = np.isfinite(both[featureName])
    both = both[legitData]
    # Remove rows with ? as features
    both = both.dropna(axis='rows')
    
    # Create training data
    training = both.copy()
    del training[featureName]
    
    # Create truth data
    truth = both[featureName]
    
    return training, truth


def add_in_dist_btw_landmarks(training):
    nrows, ncols = training.shape

    # Create labels for features
    labels = []
    for i in range(0,28):
        for j in range(i+1, 28):
            labels.append("distLandmark"+str(i+1)+"_"+str(j+1))
            
    # Create distance matrix (data)
    distanceMatrix = np.zeros((nrows, len(labels)))
    for i in range(0,nrows):
        xvals = training.ix[i,[0,28,1,29,2,30,3,31,4,32,5,33,6,34,7,35,8,36,9,37,10,38,
                               11,39,12,40,13,41,14,42,15,43,16,44,17,45,18,46,19,47,20,48,
                               21,49,22,50,23,51,24,52,25,53,26,54,27,55]].reshape(28,2)

        dists = scipy.spatial.distance.pdist(xvals, 'euclidean')
        distanceMatrix[i,:] = dists

    # Include new features in the training data
    for i in range(0,len(labels)):
        training[labels[i]] = distanceMatrix[:,i]
    
    return training

def add_in_temporal_dist(training, truth):
    nrows, ncols = training.shape
    columnLabels = training.columns.values
    training["truth"] = truth
    
    # Create new features for the filename and frame ID and sort by them
    splitRowLabels = [x.split(".csv_") for x in training.index]
    training["filename"] = [x[0] for x in splitRowLabels]
    training["frameId"] = [int(x[1]) for x in splitRowLabels]
    training = training.sort(["filename","frameId"])

    # Create new columns for the time-offset features
    for i in range(0, ncols):
        training["shiftTo_" + columnLabels[i]] = np.nan
        training["shiftFrom_" + columnLabels[i]] = np.nan

    # Calculate the new features
    filenames = pd.unique(training["filename"].values)
    for fn in filenames:
        fnIndicator = (training["filename"] == fn)
        for i in range(0, ncols):
            if (columnLabels[i] == "filename" or columnLabels[i] == "frameId" or columnLabels[i] == "truth"):
                continue
            training["shiftTo_" + columnLabels[i]][fnIndicator] = training.ix[fnIndicator,i].diff()
            training["shiftFrom_" + columnLabels[i]][fnIndicator] = -1*training.ix[fnIndicator,i].diff(periods = -1)

    # Remove the rows that appear as part of the previous or next calculation
    allowableFrameIds = range(2,500,3) #500 is magic number for faster processing (more than the max frames we have)
    training = training[training["frameId"].isin(allowableFrameIds)]
    training = training.dropna(axis='rows')

    # Save the sorted and modded truth data, then remove the redundant features
    truth = training["truth"]
    del training["filename"]
    del training["frameId"]
    del training["truth"]
    
    # Return revised dataset
    return training, truth
	
def run_evaluation(featureOfInterest):
	training, truth = import_and_split_intraface(featureOfInterest)
	trainingWithDist = add_in_dist_btw_landmarks(training.copy())
	trainingWithAllDist, truthWithAllDist = add_in_temporal_dist(trainingWithDist.copy(), truth.copy())

	run_labels = []
	performance_results = []

	models = {"Logistic regression": linear_model.LogisticRegression(), 
			  "SVM - RBF Kernel": svm.SVC(kernel='rbf') }
	for modelId in models:
		print "####################"
		print modelId
		print "####################"
		model = models[modelId]
		
		res = eval_correct(model, training.copy(), truth.copy())
		performance_results.extend([res])
		run_labels.append(modelId + ", no additional features")
		
		res = eval_correct(model, trainingWithDist.copy(), truth.copy())
		performance_results.extend([res])
		run_labels.append(modelId + ", with same-timestep distance features")

		res = eval_correct(model, trainingWithAllDist.copy(), truthWithAllDist.copy())
		performance_results.extend([res])
		run_labels.append(modelId + ", with same-timestep distance features and cross-timestep change distance features")

	print run_labels
	for run in performance_results:
		print "%0.2f %0.2f %d %d %d %d %d" % (run[0], run[1], run[2], run[3], run[4], run[5], run[6])

		

# Auto-launch as script
run_evaluation("wh question => (whq)")