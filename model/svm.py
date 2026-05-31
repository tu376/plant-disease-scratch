from sklearn.svm import SVC
class SVMModel:

    def __init__(
        self,
        C=10.0,
        kernel="rbf",
        gamma="scale",
        probability=False
    ):

        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=probability
        )

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        if not hasattr(self.model, "predict_proba"):
            raise ValueError(
                "probability=False when creating SVMModel"
            )

        return self.model.predict_proba(X)