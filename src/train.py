from sklearn.linear_model import LogisticRegression
def train_model(X, y):
    m = LogisticRegression(max_iter=500)
    m.fit(X, y)
    return m
