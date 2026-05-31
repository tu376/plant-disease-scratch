from xgboost import XGBClassifier


class XGBoostModel:

    def __init__(
        self,
        num_classes,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1
    ):

        self.model = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist",
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)