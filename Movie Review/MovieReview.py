import NaiveBayes
import csv
import math
def createCsvFile(entirePositive,entireNegative, positiveOccurence, negativeOccurence):
    
    totalnumPositiveWords = 0
    totalnumNegativeWords = 0

    for line in entirePositive.split('\n'):
        words = line.split(' ')
        totalnumPositiveWords = totalnumPositiveWords + len(words)

    for line in entireNegative.split('\n'):
        words = line.split(' ')
        totalnumNegativeWords = totalnumNegativeWords + len(words)
    
    with open('reviews.csv','w') as csvfile:
        review_writer = csv.writer(csvfile)

        sumScore = 0
        for positiveLine in entirePositive.split('\n'):
            reviewScore = 0
            row1 = []
            for word in positiveLine.split(' '):
                if(not(word == '') and (positiveOccurence.get(word,-1) != -1) and (negativeOccurence.get(word,-1) != -1)):
                    score = getScore(positiveOccurence[word],negativeOccurence[word],totalnumPositiveWords,totalnumNegativeWords)
                    reviewScore = reviewScore + score
                else:
                    score = 0;
            row1.append(reviewScore)
            row1.append(len(positiveLine))
            row1.append(1)

            review_writer.writerow(row1);

        for negativeLine in entireNegative.split('\n'):
            reviewScore = 0
            row1 = []
            for word in negativeLine.split(' '):
                if(not(word == '') and (positiveOccurence.get(word,-1) != -1) and (negativeOccurence.get(word,-1) != -1)):
                    score = getScore(positiveOccurence[word],negativeOccurence[word],totalnumPositiveWords,totalnumNegativeWords)
                    reviewScore = reviewScore + score
            row1.append(reviewScore)
            row1.append(len(negativeLine))
            row1.append(0)

            review_writer.writerow(row1);
        
            
            

        

def getScore(positiveOccurence,negativeOccurence,positivewords,negativewords):
    score = math.log(((positiveOccurence+1)/positivewords)*negativewords/(negativeOccurence+1))
    return score
        
    
def main():

    
    positiveWordOccurences = dict()

    negativeWordOccurences = dict()
    entirePositiveString = ''
    entireNegativeString = ''
    with open('positiveFile.txt','r') as myfile:#for the positive file
        line = myfile.readline()
        for word in line.split(' '):
            if(positiveWordOccurences.get(word,-1) != -1):
                positiveWordOccurences[word] = positiveWordOccurences[word] + 1
            else:
                positiveWordOccurences[word] = 1

            for negword in entireNegativeString.split(' '):
                if(word == negword):
                    if(negativeWordOccurences.get(word,-1) != -1):
                        negativeWordOccurences[word] = negativeWordOccurences[word] + 1
                    else:
                        negativeWordOccurences[word] = 1
        

    file2 = open('positiveFile.txt','r');
    entirePositiveString = file2.read()

    file2 = open('negativeFile.txt','r');
    entireNegativeString = file2.read()

                    

    with open("positiveFile.txt",'r') as myfile:
        line = myfile.readline();
        for word in line.split(' '):
            if(negativeWordOccurences.get(word,-1) != -1):
                negativeWordOccurences[word] = negativeWordOccurences[word] + 1
            else:
                negativeWordOccurences[word] = 1

            ## get the occurence of words from the engative review in the positive one
            for pword in entirePositiveString.split(' '):
                if(word == pword):
                    if(positiveWordOccurences.get(word,-1) != -1):
                        positiveWordOccurences[word] = positiveWordOccurences[word] + 1
                    else:
                        positiveWordOccurences[word] = 1
    createCsvFile(entirePositiveString, entireNegativeString,positiveWordOccurences, negativeWordOccurences)
                    

    filename = 'reviews.csv'
    splitRatio = 0.7
    
    #create a NaiveBayes object
    dataset = NaiveBayes.loadCsv(filename)
    trainingSet, testSet = NaiveBayes.splitData(dataset, splitRatio)
    splitRatio = 0.5
    testSet, developementSet = NaiveBayes.splitData(testSet, splitRatio)

    summaries = NaiveBayes.summarizeByClass(trainingSet)

    predictions = NaiveBayes.getPredictions(summaries, testSet)
    print('Predictions: {0}%'.format(predictions))
    accuracy = NaiveBayes.getAccuracy(testSet, predictions)
    print('Accuracy: {0}%'.format(accuracy))
    
                
    




main()
