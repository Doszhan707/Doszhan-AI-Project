from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ai_modules.algorithms import (
    run_linear, run_logistic, run_tree, run_rf,
    run_knn, run_nb, run_svm, run_gb,
    run_kmeans, run_pca, run_fpg, run_cv
)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = {}
    accuracies = {}
    clusters = []
    pca_result = []
    fpg_result = []
    cv_result = None

    if request.method == 'POST':
        # Input values
        education = request.form['education']
        interests = request.form.getlist('interests')
        skills = request.form.getlist('skills')
        preference = request.form['preference']
        personality = request.form['personality']

        # Feature vector
        feature_vector = [len(education), len(interests), len(skills),
                          sum(ord(c) for c in preference) % 10,
                          sum(ord(c) for c in personality) % 10]

        X = np.random.normal(loc=feature_vector, scale=1.0, size=(100, 5))
        y = np.random.choice(['Data Scientist', 'UX Designer', 'Lawyer', 'Engineer'], 100)
        y_encoded = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

        # Supervised
        predictions['Linear Regression'], accuracies['Linear Regression'] = run_linear(X_train, y_train, X_test, y_test)
        predictions['Logistic Regression'], accuracies['Logistic Regression'] = run_logistic(X_train, y_train, X_test, y_test)
        predictions['Decision Tree'], accuracies['Decision Tree'] = run_tree(X_train, y_train, X_test, y_test)
        predictions['Random Forest'], accuracies['Random Forest'] = run_rf(X_train, y_train, X_test, y_test)
        predictions['KNN'], accuracies['KNN'] = run_knn(X_train, y_train, X_test, y_test)
        predictions['Naive Bayes'], accuracies['Naive Bayes'] = run_nb(X_train, y_train, X_test, y_test)
        predictions['SVM'], accuracies['SVM'] = run_svm(X_train, y_train, X_test, y_test)
        predictions['Gradient Boosting'], accuracies['Gradient Boosting'] = run_gb(X_train, y_train, X_test, y_test)

        # Unsupervised
        clusters = list(map(int, run_kmeans(X)))
        pca_data = run_pca(X)
        pca_result = [[round(x, 4), round(y, 4)] for x, y in pca_data]

        # Association rules
        transactions = [interests + skills]
        fpg_result = [
            {'support': round(item['support'], 2), 'items': list(item['itemsets'])}
            for item in run_fpg(transactions)
        ]

        # Computer Vision
        cv_result = run_cv('static/images/sample.png')

    return render_template('index.html',
                           predictions=predictions,
                           accuracies=accuracies,
                           clusters=clusters,
                           pca_result=pca_result,
                           fpg_result=fpg_result,
                           cv_result=cv_result)

if __name__ == '__main__':
    app.run(debug=True)
