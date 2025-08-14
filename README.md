# ElevateLabsInternshipTask7
# 🧠 Breast Cancer Classification using Support Vector Machines (SVM)

This project is part of my **AI & ML Internship - Task 7**.  
The objective was to implement **Support Vector Machines** for both **linear** and **non-linear** classification on the **Breast Cancer dataset**.

---

## 📌 Objective
- Understand **margin maximization** in SVM.
- Explore the **kernel trick** for non-linear data.
- Tune important hyperparameters like **C** and **gamma**.
- Evaluate model performance with **cross-validation**.
- Visualize decision boundaries in 2D space.

---

## 📂 Dataset
I used the **Breast Cancer Wisconsin Dataset** (`breast-cancer.csv`).  
The dataset contains:
- **Diagnosis** column (`M` = Malignant, `B` = Benign) — our target variable.
- Multiple **cell nucleus features** (mean, standard error, and worst values).

---

## ⚙️ Steps Performed

### 1️⃣ Data Preparation
- Removed **`id`** column (not useful for prediction).
- Encoded `diagnosis` as:
  - `M` → 1 (Malignant)
  - `B` → 0 (Benign)
- Standardized features using **`StandardScaler`**.

### 2️⃣ Model Training
- **Linear SVM**: Straightforward hyperplane for separation.
- **RBF SVM**: Used **`GridSearchCV`** to find best `C` and `gamma`.

### 3️⃣ Evaluation
- Accuracy, Confusion Matrix, Classification Report.
- **5-fold Cross Validation** for reliability.

### 4️⃣ Visualization
- Applied **PCA** to reduce dimensions to 2D.
- Plotted decision boundaries for:
  - Linear Kernel
  - RBF Kernel

---

## 📊 Results

| Model      | Accuracy | Best Params (if applicable) |
|------------|----------|-----------------------------|
| Linear SVM | ~97%     | C = 1                       |
| RBF SVM    | ~98%     | C and gamma tuned via GridSearch |

✅ **Observation**:  
- Linear kernel already performs well because the data is mostly linearly separable.
- RBF kernel can capture slightly more complex boundaries.

---

## 📸 Visualizations
### Linear SVM Decision Boundary (PCA 2D)
*Shows the hyperplane separating malignant and benign cases in reduced dimensions.*

### RBF SVM Decision Boundary (PCA 2D)
*Shows more flexible curves capturing subtle data patterns.*

---

## 🛠 Technologies Used
- Python
- pandas, NumPy
- scikit-learn
- matplotlib

---

## 🚀 How to Run
1. Clone this repo:
   ```bash
   git clone <your_repo_link>
   cd <your_repo_folder>
