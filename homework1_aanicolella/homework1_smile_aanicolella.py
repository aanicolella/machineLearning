import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.mean(y == yhat)

def measureAccuracyOfPredictors (predictors, X, y):
    
    sumMatrix = np.zeros(y.shape)
    
    # for each image in image set, run ensemble for prediction
    for pair in predictors:
        r1,c1,r2,c2 = pair
        diff = X[:,r1,c1] - X[:,r2,c2]
        diff[diff < 0] = 0
        diff[diff > 0] = 1
        sumMatrix += diff
        
    meanMat = np.divide(sumMatrix,len(predictors))
    meanMat[meanMat < 0] = 0
    meanMat[meanMat > 0] = 1
    
    # Compute and return prediction accuracy 
    return fPC(y, meanMat)
    

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    
    predictors = []
    topAcc = []
    for i in range(0,5):
        topAcc = 0.0
        topFeat = None
        for r1 in range(0,24):
            for c1 in range(0,24):
                for r2 in range(0,24):
                    for c2 in range(0,24):
                        
                        # If pixel one and pixel two same, skip
                        if (r1,c1)  == (r2,c2):
                            continue
                        
                        currLoc = (r1,c1,r2,c2)
                        # If pixel pair already identified/investigated, skip
                        if currLoc in predictors:
                            continue
                        
                        currAcc = measureAccuracyOfPredictors(predictors +  list((currLoc,)), trainingFaces,trainingLabels)
                        
                        # Update topAcc/topFeat if current feature has higher score
                        if currAcc > topAcc:
                            topAcc = currAcc
                            topFeat = currLoc
        
        # Append top features for m = i before i++
        predictors.append(topFeat)
            
    return predictors
        
def visFeatures(predictors, testingFaces):
    
        im = testingFaces[0,:,:]    
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')        
        for pair in predictors:
            r1,c1,r2,c2 = pair
            pixel = patches.Rectangle((c1 - 0.5, r1 - 0.5),1,1,linewidth=2,edgecolor='b',facecolor='none')
            ax.add_patch(pixel)
    		# Show r2,c2
            pixel = patches.Rectangle((c2 - 0.5, r2 - 0.5),1,1,linewidth=2,edgecolor='r',facecolor='none')
            ax.add_patch(pixel)
        # Display merged result    
        fig.show()

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

def trainChooChoo(trainingFaces, trainingLabels, testingFaces, testingLabels):
    
    sampleNums = [400,800,1200,1600,2000]
    predictors = []
    print("n     trainingAccuracy     testingAccuracy")
    for sample in sampleNums:
        # Run stepwiseRegression on sample subset
        predictors = stepwiseRegression(trainingFaces[:sample],trainingLabels[:sample],testingFaces,testingLabels)
        
        # Measure training and testing accuracy
        trainAccuracy = measureAccuracyOfPredictors(predictors, trainingFaces, trainingLabels)
        testAccuracy = measureAccuracyOfPredictors(predictors, testingFaces, testingLabels)
        
        # Print formatted results
        print("{}      {}                {}".format(sample, trainAccuracy, testAccuracy))   

        
    # Visualize learned features
    visFeatures(predictors, trainingFaces)

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    trainChooChoo(trainingFaces, trainingLabels, testingFaces, testingLabels)
