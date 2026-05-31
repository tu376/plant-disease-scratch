from sklearn.neighbors import KNeighborsClassifier

def build_knn():

    model = KNeighborsClassifier(
        n_neighbors=5
    )
    return model