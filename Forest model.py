def predict_forest():
    forest = RandomForestClassifier(n_estimators=100)
    result = forest.fit(vectorize_that_shit())


    return result