# SiteSignal - Phishing Website Detection Using Machine Learning

**AIMLZG565 - Machine Learning · Assignment 2**

## a. Problem Statement

Phishing websites try to collect personal information by making users believe that a fake website is genuine. In this assignment, I have used machine learning to classify websites as phishing or legitimate based on their website features.

I trained and compared five classification models on the same dataset and test data. I used Accuracy, AUC, Precision, Recall, F1 Score and MCC to compare their performance. I also developed a Streamlit app where a user can upload labelled test data, select a trained model and view its evaluation results.

## b. Dataset Description

- **Dataset:** Phishing Websites
- **Source:** UCI Machine Learning Repository (Dataset ID 327)
- **Dataset link:** https://archive.ics.uci.edu/dataset/327/phishing
- **Total records:** 11,055
- **Input features:** 30
- **Problem type:** Binary classification
- **Example features:** URL length, having an IP address, SSL final state, age of domain, URL of anchor and Google index.
- **Target column in this project:** `target`
  - `1` = Phishing website
  - `0` = Legitimate website
- **Class distribution:**
  - Phishing (1): 4,898 records (44.31%)
  - Legitimate (0): 6,157 records (55.69%)
- **Class balance:** The dataset is mildly imbalanced, with slightly more legitimate websites than phishing websites. However, both classes are reasonably well represented.
- **Data split:** 80% training (8,844 records) and 20% testing (2,211 records), using a stratified split with `random_state=565`.

The original dataset uses `-1` for phishing and `1` for legitimate websites. I changed these target labels to `1` for phishing and `0` for legitimate in my project. The 30 input features were kept as provided in the dataset.

The same training and test split was used for all five models so that their results could be compared fairly. For Precision, Recall, F1 Score and AUC, phishing (`target=1`) is treated as the positive class.

## c. GitHub Repository Link

**GitHub Repository:** https://github.com/mechaitanyashintre/2026AJ05716-ML-Assignment-2

**Live Streamlit App:** https://2026aj05716-ml-assignment-2.streamlit.app/

## d. Models Used and Evaluation Results

I implemented the following five classification models:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM, RBF kernel)

I used feature scaling for Logistic Regression, KNN and SVM. For KNN, I used five neighbors. The preprocessing steps and the trained classifiers were saved together as pipelines. All the results below were calculated on the same **2,211 test records**.

**Note on KNN consistency:** During testing, the original KNN implementation gave slightly different evaluation results between the local testing environment and Streamlit Cloud. This can occur when multiple training records are equally distant from a test record. The [scikit-learn KNeighborsClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) notes that when the kth and (k+1)th neighbors are equally distant but have different labels, the result can depend on training-data order. To keep the neighbor selection consistent, I used stable sorting with a fixed tie-breaking rule: when distances are equal, the training row appearing earlier is selected first. The updated KNN implementation was checked locally and on Streamlit Cloud.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9281 | 0.9752 | 0.9419 | 0.8929 | 0.9167 | 0.8545 |
| Decision Tree | 0.9263 | 0.9825 | 0.9323 | 0.8990 | 0.9153 | 0.8505 |
| KNN | 0.9435 | 0.9849 | 0.9358 | 0.9367 | 0.9363 | 0.8855 |
| Naive Bayes | 0.5952 | 0.9630 | 0.5228 | 0.9959 | 0.6856 | 0.3729 |
| Support Vector Machine | 0.9512 | 0.9887 | 0.9668 | 0.9214 | 0.9436 | 0.9014 |

### Observations About Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression gave good results with 92.81% accuracy and an F1 Score of 0.9167. Its precision (0.9419) was higher than its recall (0.8929). It correctly detected 875 phishing websites, but missed 105 phishing websites and wrongly marked 54 legitimate websites as phishing. |
| Decision Tree | Decision Tree gave 92.63% accuracy and an F1 Score of 0.9153, which was close to Logistic Regression. Its AUC (0.9825) was higher than Logistic Regression's AUC (0.9752), but its F1 and MCC were slightly lower. It missed 99 phishing websites and wrongly flagged 64 legitimate websites as phishing. |
| KNN | KNN performed better than Logistic Regression and Decision Tree, with 94.35% accuracy and an F1 Score of 0.9363. Its recall (0.9367) was higher than SVM's recall. Out of 980 phishing websites in the test data, KNN correctly detected 918 and missed 62, but it also wrongly flagged 63 legitimate websites. |
| Naive Bayes | Naive Bayes had the highest recall (0.9959), correctly detecting 976 out of 980 phishing websites. However, its precision was only 0.5228 and its accuracy was 59.52%, because it incorrectly marked 891 legitimate websites as phishing. Even though its AUC was high (0.9630), its overall classification performance at the chosen threshold was weaker than the other models. |
| Support Vector Machine | SVM gave the highest accuracy (95.12%), AUC (0.9887), precision (0.9668), F1 Score (0.9436) and MCC (0.9014). It wrongly flagged only 31 legitimate websites as phishing, the fewest among all five models. It missed 77 phishing websites, slightly more than KNN, but gave the best overall balance of results on this test data. |
| **Overall Winner for My Dataset** | **Support Vector Machine (SVM)** is the overall winner because it achieved the highest F1 Score and MCC, along with the highest accuracy, precision and AUC. For this test data, it gave a good balance between detecting phishing websites and avoiding incorrect phishing warnings for legitimate websites. |

### Project Files

| File / Folder | Purpose |
|---|---|
| `download_dataset.py` | Loads the UCI dataset and prepares `dataset.csv`. |
| `train_models.py` | Trains the five models, evaluates them and saves the results. |
| `stable_knn.py` | KNN classifier with consistent tie-breaking for equally distant training rows. |
| `verify_project.py` | Checks the saved models and recalculates their evaluation metrics. |
| `app.py` | Streamlit application for uploading labelled test data and comparing saved models. |
| `requirements.txt` | Python libraries required for the project. |
| `dataset.csv` | Prepared dataset containing 30 features and the target column. |
| `test_data.csv` | Labelled test data used for evaluation. |
| `model/` | Five saved trained models (`.joblib`) and model information. |
| `results/` | Evaluation metrics saved in CSV and Markdown formats. |

### How to Run

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

Run the Streamlit app using the already saved models:

```bash
python -m streamlit run app.py
```

To reproduce the training and evaluation results, run:

```bash
python download_dataset.py
python train_models.py
python verify_project.py
```

In the Streamlit app, upload `test_data.csv` and select a model to view its metrics, confusion matrix and comparison results. The uploaded file must contain all 30 input features and the `target` column. **The app loads the saved models and does not retrain them on uploaded test data.**

This project evaluates websites using already extracted features. It does not check a website directly from its URL.
