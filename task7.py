import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv("breast-cancer.csv")

# 2. Drop unnecessary columns
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# 3. Encode target variable
if df['diagnosis'].dtype == 'object':
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 4. Features & labels
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# 5. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 7. Linear SVM
linear_svm = SVC(kernel='linear', C=1)
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

print("=== Linear SVM ===")
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

# 8. RBF SVM with hyperparameter tuning
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
rbf_svm = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
rbf_svm.fit(X_train, y_train)

best_rbf = rbf_svm.best_estimator_
y_pred_rbf = best_rbf.predict(X_test)

print("=== RBF SVM ===")
print("Best Params:", rbf_svm.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))

# 9. Cross-validation score for best RBF model
cv_scores = cross_val_score(best_rbf, X_scaled, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# 10. PCA for visualization (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Retrain models in 2D space
linear_svm_2d = SVC(kernel='linear', C=1)
linear_svm_2d.fit(X_pca, y)

rbf_svm_2d = SVC(kernel='rbf', C=rbf_svm.best_params_['C'], gamma=rbf_svm.best_params_['gamma'])
rbf_svm_2d.fit(X_pca, y)

# Function to plot decision boundary
def plot_decision_boundary(model, X, y, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.show()

# 11. Plot both decision boundaries
plot_decision_boundary(linear_svm_2d, X_pca, y, "Linear SVM Decision Boundary (PCA 2D)")
plot_decision_boundary(rbf_svm_2d, X_pca, y, "RBF SVM Decision Boundary (PCA 2D)")
