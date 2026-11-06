"""app.py - Streamlit interface for comparing the five saved classifiers.

Run after training and verification: python -m streamlit run app.py
Upload test_data.csv (30 features and target), then choose a classifier.
This app loads the saved .joblib pipelines, predicts and calculates metrics.
It does not download the dataset or retrain models on uploaded records.
"""

from pathlib import Path  # Find saved files and set up the chart axis.
import json
import math

import altair as alt  # Create the F1 comparison chart.
import joblib  # Load saved models.
import pandas as pd  # Work with uploaded CSV files.
import streamlit as st  # Build the Streamlit page.
from sklearn.metrics import (  # Import metrics for the uploaded test data.
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parent  # Current project folder.
MODEL_DIR = PROJECT_DIR / 'model'  # Location of trained model files.

st.set_page_config(  # Set the tab title and use a wide page layout.
    page_title='SiteSignal | Phishing Detection Lab',
    page_icon='🛡️',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# Style the page with the dark theme.
# Only appearance is customized here. ML algorithms and formulas are unchanged.
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 90% 2%, #122943 0, transparent 35%),
                    radial-gradient(circle at 3% 20%, #102637 0, transparent 30%),
                    #0B1220;
        color: #E6EDF6;
    }
    .block-container { max-width: 1320px; padding-top: 5.0rem !important; padding-bottom: 3rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(160deg, #17273D, #111D30);
        border: 1px solid #29415A;
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 108px;
    }
    div[data-testid="stMetricLabel"] { color: #A9BDD1; font-size: 0.90rem; }
    div[data-testid="stMetricValue"] { color: #F1F7FE; }
    div[data-testid="stFileUploader"], div[data-testid="stSelectbox"] { border-radius: 12px; }
    div[data-testid="stDataFrame"] { border: 1px solid #2A4056; border-radius: 12px; overflow: hidden; }
    h2, h3 { letter-spacing: -0.025em; }
    .hero-panel {
        border: 1px solid #32536D;
        border-radius: 18px;
        padding: 28px 32px;
        background: linear-gradient(112deg, #152F49 0%, #14233A 52%, #13283D 100%);
        margin-bottom: 12px;
    }
    .hero-eyebrow { color: #61D6CB; font-size: 0.79rem; letter-spacing: 0.13em; font-weight: 750; }
    .hero-title { color: #F3F9FF; font-size: 2.1rem; font-weight: 780; letter-spacing: -0.04em; margin: 8px 0; line-height: 1.15; }
    .hero-description { color: #B8CCDF; font-size: 0.96rem; max-width: 880px; line-height: 1.6; }
    .student-meta { color: #C8E0F0; font-size: 0.88rem; margin: 12px 0 8px; line-height: 1.6; }
    .hero-pill {
        color: #8AE9D8; border: 1px solid #2E746C;
        background-color: #123C42; border-radius: 999px;
        font-size: 0.78rem; font-weight: 650;
        display: inline-block; padding: 5px 12px; margin-top: 16px;
    }
    .section-note { color: #AFC2D5; font-size: 0.88rem; margin-bottom: 12px; }
    hr { border-color: #294159 !important; }
    @media (max-width: 780px) {
        .hero-panel { padding: 22px; }
        .hero-title { font-size: 1.7rem; }
    }
</style>
""", unsafe_allow_html=True)

# Show the project title and assignment details.
st.markdown("""
<div class="hero-panel">
  <div class="hero-eyebrow"><strong>AIMLZG565 - Machine Learning</strong> · Assignment 2</div>
  <div class="hero-title">🛡️ SiteSignal | Phishing Website Detection</div>
  <div class="hero-description">
    Compare five previously trained ML classifiers using the same labelled test CSV.
    All metrics are calculated from the uploaded test records.
  </div>
  <span class="hero-pill">SAVED MODELS · TEST DATA ONLY · NO RETRAINING</span>
</div>
""", unsafe_allow_html=True)

config_path = MODEL_DIR / 'model_info.json'  # Read model details saved during training.
if not config_path.exists():  # Stop if training outputs are missing.
    st.error('Saved models are missing. Run download_dataset.py and train_models.py first.')
    st.stop()

config = json.loads(config_path.read_text(encoding='utf-8'))  # Read model names, files and input columns.
feature_columns = config['feature_names']  # Use features in their original training order.
model_files = config['model_files']  # Names of the five saved model files.

# Show dataset details on the main page.
# Keep the required dataset information in the main page; no sidebar.
st.markdown(
    f"**Dataset:** UCI Phishing Websites (ID 327) &nbsp; · &nbsp; "
    f"**Instances:** 11,055 &nbsp; · &nbsp; "
    f"**Input features:** {len(feature_columns)} &nbsp; · &nbsp; "
    f"**Held-out test records:** {config['test_rows']:,}"
)
st.caption('Labels (Target Column): 0 = legitimate, 1 = phishing (positive class). '
           f"80/20 stratified split, random state {config['random_state']}. This app does not scan live URLs.")
st.divider()  # Separate the dataset details from the upload section.

st.markdown('### 01 · Upload test CSV & select model')  # Upload and model selection section.
# Explain what columns are needed.
st.markdown('<div class="section-note">Upload a test CSV containing all 30 feature columns and the <code>target</code> column.</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader('Test CSV upload (30 features + target)', type=['csv'])  # Accept CSV files only.
example_path = PROJECT_DIR / 'test_data.csv'  # Use the saved test file as an example.
if example_path.exists():  # Offer the generated test CSV as a download.
    st.download_button(
        'Download reference test CSV',
        data=example_path.read_bytes(),
        file_name='test_data.csv',
        mime='text/csv',
        width='content',
    )

preferred = 'Support Vector Machine'  # Select SVM initially.
chosen_model = st.selectbox(  # Show the five classifiers in a dropdown.
    'Classifier (Model selection)',
    options=list(model_files),
    index=list(model_files).index(preferred) if preferred in model_files else 0,
)

if uploaded_file is None:  # Wait for the CSV upload before showing results.
    st.info('Upload the generated test_data.csv to see metrics, confusion matrix and all-model comparison.')
    st.divider()  # Show the footer even before a CSV is uploaded.
    st.caption('NAME: CHAITANYA PRAMOD SHINTRE')
    st.stop()

if uploaded_file.size > 5 * 1024 * 1024:  # Limit the upload to 5 MB.
    st.error('Upload only test data (maximum 5 MB).')
    st.stop()
try:  # Read the uploaded CSV and report errors.
    uploaded_data = pd.read_csv(uploaded_file)  # Load the uploaded rows into a table.
except Exception as error:
    st.error('Could not read CSV: ' + str(error))
    st.stop()

# Check for all 30 features and the target column.
missing_columns = [col for col in feature_columns + [config['target_column']] if col not in uploaded_data.columns]
if missing_columns:  # List any missing columns.
    st.error('Missing columns: ' + ', '.join(missing_columns))
    st.stop()
if uploaded_data.empty:  # Do not evaluate an empty file.
    st.error('The uploaded file has no rows.')
    st.stop()

try:  # Convert the inputs and target to numeric values.
    X_test = uploaded_data[feature_columns].apply(pd.to_numeric, errors='raise')  # Get features from the uploaded CSV.
    y_test = pd.to_numeric(uploaded_data[config['target_column']], errors='raise')  # Get the actual labels from the uploaded CSV.
except Exception as error:
    st.error('Features and target must be numeric: ' + str(error))
    st.stop()

if X_test.isna().any().any():  # Check that no feature values are missing.
    st.error('Missing feature values found. Please use a clean test CSV.')
    st.stop()
if not X_test.isin([-1, 0, 1]).all().all():  # Only -1, 0 and 1 are allowed for the UCI features.
    st.error('UCI input features must contain only -1, 0 or 1.')
    st.stop()
if y_test.isna().any() or not y_test.isin([0, 1]).all():  # Target must contain valid class labels.
    st.error('Target values must be 0 (legitimate) or 1 (phishing).')
    st.stop()
y_test = y_test.astype(int)  # Use integer targets for scoring.


@st.cache_resource  # Cache models to avoid reloading them on every rerun.
def load_model(filename):  # Load a saved model rather than training it again.
    # Filename is taken from our own saved configuration, not from the uploaded CSV.
    return joblib.load(MODEL_DIR / filename)  # Load the chosen .joblib pipeline.


def compute_scores(model, inputs, actual):  # Score a model on the uploaded records.
    # Same evaluation formulas as train_models.py; no fitting here.
    predicted = model.predict(inputs)  # Predict the class of each uploaded record.
    phishing_index = list(model.classes_).index(1)  # Find the probability column for phishing.
    phishing_probabilities = model.predict_proba(inputs)[:, phishing_index]  # Use phishing probabilities for AUC.
    # AUC needs both actual classes in the upload.
    auc = roc_auc_score(actual, phishing_probabilities) if actual.nunique() == 2 else float('nan')
    scores = {  # Calculate all six assignment metrics.
        'Accuracy': accuracy_score(actual, predicted),  # Fraction of correct predictions.
        'AUC': auc,  # ROC-AUC value, or N/A for single-class uploads.
        # Correct phishing alerts among all phishing alerts.
        'Precision': precision_score(actual, predicted, zero_division=0),
        'Recall': recall_score(actual, predicted, zero_division=0),  # Detected phishing among actual phishing.
        'F1': f1_score(actual, predicted, zero_division=0),  # Precision-recall balance.
        'MCC': matthews_corrcoef(actual, predicted),  # Balanced score using TN, FP, FN and TP.
    }
    return scores, predicted  # Return scores and predicted labels.


# Use the same uploaded records for all five models.
# All five models are measured on the exact same uploaded test records.
model_results = []  # Collect a score row for each model.
chosen_predictions = None  # Keep the chosen model's predictions for the confusion matrix.
chosen_metrics = None  # Keep the chosen model's six scores.
try:  # Catch model-loading and scoring errors.
    # Show a spinner only while saved models are being loaded and evaluated.
    # No artificial progress or delays are added.
    # Show a spinner while results are calculated.
    with st.spinner('Evaluating the five saved models on your uploaded test data...'):
        for name, filename in model_files.items():  # Evaluate each saved model.
            model = load_model(filename)  # Load the fitted pipeline.
            model_scores, predictions = compute_scores(model, X_test, y_test)  # Score it on the uploaded test data.
            model_results.append({'ML Model Name': name, **model_scores})  # Add its results to the comparison table.
            if name == chosen_model:  # Store details for the dropdown selection.
                chosen_predictions = predictions
                chosen_metrics = model_scores
except Exception as error:
    st.error('Unable to evaluate saved models: ' + str(error))
    st.stop()

# Show how many test rows were processed.
st.success(f'✓ {len(X_test):,} test records evaluated successfully — trained models loaded, no retraining.')
st.divider()
st.markdown('### 02 · Selected model results')  # Display the six scores for the selected model.
st.caption(f'Currently selected: **{chosen_model}** · Positive class: phishing (1)')
metric_names = list(chosen_metrics)  # Names of the six scores.
for group_start in (0, 3):  # Split the six metrics into two rows of three.
    cols = st.columns(3)  # Create three columns for this row.
    for i, col in enumerate(cols, start=group_start):  # Place each score in the correct column.
        metric_name = metric_names[i]
        metric_value = chosen_metrics[metric_name]
        # Display N/A when AUC cannot be calculated.
        col.metric(metric_name, 'N/A' if pd.isna(metric_value) else f'{metric_value:.4f}')
if pd.isna(chosen_metrics['AUC']):  # Explain why AUC can be unavailable.
    st.caption('AUC is undefined because the uploaded CSV has only one target class.')

st.divider()
st.markdown('### 03 · Confusion matrix')  # Confusion matrix section.
cm = confusion_matrix(y_test, chosen_predictions, labels=[0, 1])  # Calculate the 2-by-2 confusion matrix.
tn, fp, fn, tp = cm.ravel()  # TN, FP, FN and TP are read from the matrix.
st.dataframe(  # Display the confusion matrix as a table.
    pd.DataFrame(
        cm,
        index=['Actual legitimate (0)', 'Actual phishing (1)'],
        columns=['Predicted legitimate (0)', 'Predicted phishing (1)'],
    ),
    width='stretch',
)

# All four confusion-matrix counts on one row, correct predictions first.
tp_col, tn_col, fn_col, fp_col = st.columns(4)  # Show four confusion-matrix counters in one row.
with tp_col:  # Display the correct phishing count.
    st.metric('Correct phishing (TP)', int(tp))
with tn_col:  # Display the correct legitimate count.
    st.metric('Correct legitimate (TN)', int(tn))
with fn_col:  # Display the missed phishing count.
    st.metric('Missed phishing (FN)', int(fn))
with fp_col:  # Display the false alert count.
    st.metric('False alerts (FP)', int(fp))
st.caption('False negative: a phishing site predicted legitimate. '
           'False positive: a legitimate site predicted phishing.')
st.caption('True positive: a phishing site correctly predicted phishing. '
           'True negative: a legitimate site correctly predicted legitimate.')

# Show extra per-class results when expanded.
# Optional extra: per-class metrics without cluttering the main page.
# Keep the report collapsed initially.
with st.expander('📑 Detailed classification report (click to expand)', expanded=False):
    report = classification_report(  # Calculate per-class precision, recall and F1.
        y_test, chosen_predictions, labels=[0, 1],
        target_names=['Legitimate', 'Phishing'],
        zero_division=0, output_dict=True,
    )
    # The special "accuracy" entry has no record-count support, so omit it.
    # Accuracy is already visible in the six metrics above.
    report_rows = []  # Make the rows for the detailed report.
    for label in ['Legitimate', 'Phishing', 'macro avg', 'weighted avg']:  # Include both classes and both averages.
        values = report[label]  # Read each row from sklearn's report.
        report_rows.append({
            'Class': label.title() if label in ['macro avg', 'weighted avg'] else label,
            'Precision': f"{values['precision']:.4f}",
            'Recall': f"{values['recall']:.4f}",
            'F1 Score': f"{values['f1-score']:.4f}",
            'Support (records)': int(values['support']),  # Support is the number of actual records in the class.
        })
    st.dataframe(pd.DataFrame(report_rows), hide_index=True, width='stretch')  # Display the report as a table.
    st.caption('Accuracy is displayed above; support is the count of records for each class.')

st.divider()  # Show the full comparison below the confusion matrix.
st.markdown('### 04 · Compare all five models')
results = pd.DataFrame(model_results).set_index('ML Model Name')  # Make a table of five models and six scores.
st.dataframe(results.round(4), width='stretch')  # Display rounded values for readability.
st.caption('Results are recalculated for all five saved models on this same uploaded CSV. '
           'Precision, Recall and F1 treat phishing (1) as positive; AUC uses predicted probabilities.')

# Optional visual comparison: shown directly under the required results table.
# A horizontal, ranked chart makes closely grouped F1 scores easier to read.
st.markdown('#### 📊 F1 score comparison')  # F1 bar chart section.
f1_chart_data = results[['F1']].reset_index().sort_values('F1', ascending=False).copy()  # Sort models by highest F1 first.
f1_chart_data['Score label'] = f1_chart_data['F1'].map(lambda number: f'{number:.4f}')  # Make labels with four decimal places.
top_model_name = f1_chart_data.iloc[0]['ML Model Name']  # Find the model with the highest F1.
model_order = f1_chart_data['ML Model Name'].tolist()  # Keep that order for the bars.

# Use a clearly labelled zoomed axis (not a zero-based bar chart).
# The start is determined by the lowest F1 score so any valid uploaded CSV works.
# Choose an axis start that makes close scores visible.
axis_start = max(0.0, math.floor((f1_chart_data['F1'].min() - 0.05) * 10) / 10)
f1_chart_data['Axis start'] = axis_start  # Use the same starting point for every bar.

base = alt.Chart(f1_chart_data)  # Start building the chart.
chart_x = alt.X(  # Set the F1 score as the horizontal axis.
    'F1:Q',
    scale=alt.Scale(domain=[axis_start, 1.0], nice=False, zero=False),
    axis=alt.Axis(title='F1 score', format='.2f', tickCount=6),
)
chart_y = alt.Y(  # Set the models as the vertical labels.
    'ML Model Name:N',
    sort=model_order,
    axis=alt.Axis(title=None, labelLimit=300, labelFontSize=13, labelPadding=12),
)
bars = base.mark_bar(size=30, cornerRadiusEnd=6).encode(  # Draw the horizontal bars.
    x=chart_x,
    x2=alt.X2('Axis start:Q'),
    y=chart_y,
    color=alt.condition(  # Color the best model teal and the others blue.
        alt.datum['ML Model Name'] == top_model_name,
        alt.value('#4ADBC9'),  # teal: highest F1
        alt.value('#84BDF1'),  # light blue: other models
    ),
    tooltip=[
        alt.Tooltip('ML Model Name:N', title='Model'),
        alt.Tooltip('F1:Q', format='.4f', title='F1 score'),
    ],
)
score_labels = base.mark_text(  # Write the exact score on each bar.
    align='right', baseline='middle', dx=-12, fontSize=13,
    fontWeight='bold', color='#0B1930',
).encode(x=chart_x, y=chart_y, text=alt.Text('Score label:N'))
f1_chart = (bars + score_labels).properties(height=310).configure_view(  # Set chart size and background.
    stroke=None,
    fill='#101D2F',
).configure_axis(  # Set axis text, grid and tick colors.
    labelColor='#C6D6E8',
    titleColor='#B5CCDE',
    gridColor='#2A3B52',
    domainColor='#334A62',
    tickColor='#334A62',
    labelFontSize=12,
    titleFontSize=12,
)
st.altair_chart(f1_chart, width='stretch', theme=None)  # Show the chart in Streamlit.
st.caption(  # Explain that the chart uses a zoomed axis.
    f'F1 values range from {f1_chart_data["F1"].min():.4f} to {f1_chart_data["F1"].max():.4f}. '
    f'The axis starts at {axis_start:.2f} (zoomed to make small differences visible). '
    'Exact results for all six metrics are given in the table above.'
)


st.divider()  # Footer separator.
st.caption('NAME: CHAITANYA PRAMOD SHINTRE')  # Student name at the bottom of the page.
