import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import cv2

# === 1. Linear Regression ===
def run_linear(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds_class = np.clip(np.round(preds).astype(int), 0, np.max(y_test))
    acc = accuracy_score(y_test, preds_class) * 100
    return int(preds_class[0]), round(acc, 2)

# === 2. Logistic Regression ===
def run_logistic(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    return int(preds[0]), round(acc, 2)

# === 3. Decision Tree ===
def run_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    return int(preds[0]), round(acc, 2)

# === 4. Random Forest ===
def run_rf(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    return int(preds[0]), round(acc, 2)

# === 5. KNN ===
def run_knn(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    return int(preds[0]), round(acc, 2)

# === 6. Naive Bayes ===
def run_nb(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    return int(preds[0]), round(acc, 2)

# === 7. SVM ===
def run_svm(X_train, y_train, X_test, y_test):
    model = SVC()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    return int(preds[0]), round(acc, 2)

# === 8. Gradient Boosting ===
def run_gb(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    return int(preds[0]), round(acc, 2)

# === 9. KMeans Clustering ===
def run_kmeans(X):
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X)
    return model.labels_.tolist()

# === 10. PCA ===
def run_pca(X):
    model = PCA(n_components=2)
    result = model.fit_transform(X)
    return result.tolist()

# === 11. FP-Growth ===
def run_fpg(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    freq_items = fpgrowth(df, min_support=0.5, use_colnames=True)
    return freq_items.to_dict('records')

# === 12. Computer Vision (SIFT) ===
def run_cv(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return "Image not found."
        sift = cv2.SIFT_create()
        keypoints, _ = sift.detectAndCompute(image, None)
        return f"Detected {len(keypoints)} keypoints."
    except:
        return "Image processing failed."
