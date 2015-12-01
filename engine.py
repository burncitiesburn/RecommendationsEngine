from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS
import math
conf=SparkConf().setMaster("local").setAppName("Finding Top Movies")
sc=SparkContext(conf=conf)

numPartitions = 2
rawRatings = sc.textFile(ratingsFilename).repartition(numPartitions)
rawMovies = sc.textFile(moviesFilename)

def get_ratings_tuple(entry):
    items = entry.split('::')
    return int(items[0]), int(items[1]), float(items[2])
    
def get_movie_tuple(entry):
    items = entry.split('::')
    return int(items[0]),items[1]
    
ratingsRDD = rawRatings.map(get_ratings_tuple).cache()
moviesRDD = rawMovies.map(get_movie_tuple).cache()

movieIDsWithRatingsRDD = (ratingsRDD.map(lambda (user_id, movie_id, rating): (movie_id, [rating])).reduceByKey(lambda a,b: a+b))

def getCountsAndAverages(RatingsTuple):
    total = 0.0
    for rating in RatingsTuple[1]:
        total += rating
        
    return ( RatingsTuple[0], (len(RatingsTuple[1]), total/len(RatingsTuple[1])))

movieNameWithAvgRatingsRDD = movieIDsWithRatingsRDD.map(getCountsAndAverages)

movieNameWithAvgRatingsRDD = ( moviesRDD .join(movieNameWithAvgRatingsRDD) .map(lambda ( movieid,(name,(ratings, average)) ): (average, name, ratings)) )

def sortFunction(tuple):
    key = unicode('%.3f' % tuple[0])
    value = tuple[1]
    return (key + ' ' + value)

#this is basic recommendations

movieLimitedAndSortedByRatingRDD = ( movieNameWithAvgRatingsRDD.filter( lambda(average,name,ratings): ratings > 500).sortBy(sortFunction, ascending=False))

#write these to a file

#this is for proper recommendations
trainingRDD, validationRDD, testRDD = ratingsRDD.randomSplit([6,2,2], seed=0L)

#calculate RMSE

def computeRMSE(predictedRDD, actualRDD):
    predictedReformattedRDD = (predictedRDD.map(lambda (UserID, MovieId, Rating): ((UserID, MovieId),Rating)))
    
    actualReformattedRDD = (actualRDD.map(lambda (UserID, MovieId, Rating): ((UserID, MovieId),Rating)))
    
    squaredErrorsRDD = (predictedReformattedRDD.join(actualReformattedRDD).map(lambda(k,(a,b)): math.pow((a-b),2)))
    
    totalErrors = squaredErrorsRDD.reduce(lambda a,b: a+b)
    numRatings = squaredErrorsRDD.count()
    
    return math.sqrt(float(totalErrors)/numRatings)
validationForPredictionRDD = validationRDD.map(lambda(UserID, MovieId, Rating): (UserID,MovieId))

ranks = [4,8,12]
errors = [0,0,0]
err = 0
minError = float('inf')
bestRank = -1
bestIteration = -1

for rank in ranks:
    
    model = ALS.train(trainingRDD, rank, seed=5L, iterations =5, lambda_= 0.1)
    predictedRatingsRDD = model.predictALL(validationForPredictionRDD);
    error = computeRMSE(predictedRatingsRDD, validationRDD)
    errors[err] = error
    err += 1
    if error < minError:
        minError = error
        bestRank = rank

bestModel = ALS.train(trainingRDD, bestRank, seed=5L, interations=5, lambda_=0,1)

testForPredictingRDD = testRDD.map(lambda (UserID, MovieID, Rating): (UserID, MovieID))
predictedTestRDD = bestModel.predictAll(testForPredictingRDD)

testRMSE = computeRMSE(tesTRDD, predictedTestRDD)
#generate new recommendations for all users

userRatings = ratingsRDD.map(lambda(UserID, MovieId, Rating): (UserID,MovieId))
predictionsRDD = model.predict(userRatings).map(lambda(UserID, MovieId, rating): (UserID, MovieId),rating)
ratingsAndPredictionsRDD =  ratingsRDD.map(lambda(UserID, MovieId, rating): (UserID, MovieId),rating).join(predictionsRDD)

def apk(actual, predicted, k=10):

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)
#http://blog.5ibc.net/p/20398.html
#https://books.google.com.au/books?id=syPHBgAAQBAJ&pg=PA103&lpg=PA103&dq=recommendproducts+movielens&source=bl&ots=X9TEXS538v&sig=lCKSQTf7EIjsQVHlcE5pC2yNxLE&hl=en&sa=X&ved=0ahUKEwjmpfqRzLnJAhWCmZQKHYdtBBMQ6AEIIzAB#v=onepage&q=recommendproducts%20movielens&f=false    
#   I am here
#unratedMoviesByUserRDD = (moviesRDD.map(lambda (movieID, name): movieID).filter(lambda movieID: movieID not in [this[1] for this in myRatedMovies]).map(lambda movieID: (userID, movieID)))

#recommendationRDD = bestModel.predictAll(unratedMoviesByUserRDD)

# get only the most rated movies
#movieCountsRDD = movieIDsWithAvgRatingsRDD.map(lambda (movie_id, (ratings, average)): (movie_id, ratings)) 
#predictedRDD = predictedRatingsRDD.map(lambda (uid, movie_id, rating): (movie_id, rating)) 
#predictedWithCountsRDD= (predictedRDD.join(movieCountsRDD)) 
#ratingsWithNamesRDD = (predictedWithCountsRDD.join(moviesRDD).map(lambda (movie_id, ((pred, ratings), name)): (pred, name, ratings)).filter(lambda (pred, name, ratings): ratings > 75))

#predictedMovies = ratingsWithNamesRDD.takeOrdered(20, key=lambda X: -x[0])

sc.stop();