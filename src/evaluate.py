from sklearn.metrics import accuracy_score
def evaluate_model(m, X, y):
    return accuracy_score(y, m.predict(X))
