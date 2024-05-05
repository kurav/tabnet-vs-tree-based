import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned data
@st.cache_data
def load_data():
    return pd.read_csv('transformed_results_cleaned.csv')


st.set_page_config(page_title="Hyperparameter Tuning", layout="wide")
data = load_data()

# Navigation
page = st.sidebar.radio("Navigate", ["Home", "Model Tuning"], index=0)

if page == "Home":
    st.title("Hyperparameter Tuning - Overview")
    # getting average metrics for each model
    for model in data['Model'].unique():
        model_data = data[data['Model'] == model]
        avg_metrics = model_data[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].mean()
        st.subheader(f"Average Metrics for {model}")
        st.table(avg_metrics)

        fig, ax = plt.subplots()
        categories = list(avg_metrics.index)
        values = list(avg_metrics.values)
        ax.bar(categories, values, color='skyblue')
        plt.ylim(0, 1)
        plt.title(f"Average Performance Metrics for {model}")
        st.pyplot(fig)
elif page == "Model Tuning":
    st.title("Model-Specific Hyperparameter Tuning")

    # model selection
    model = st.sidebar.selectbox('Select Model', data['Model'].unique())

    fixed_hyperparameters = {
        'XGBoost': {
            'n_estimators': [50, 100, 150],
            'lr': [0.01, 0.05, 0.1],
            'depth': [3, 5, 7]
        },
        'TabNet': {
            'n_d_n_a': (8, 16, 24),
            'lr': (0.01, 0.02),
            'epochs': (50, 100)
        },
        'Random Forest': {
            'n_estimators': (100, 200, 300),
            'max_depth': (0, 10, 20),
            'min_samples_split': (2, 5, 10)
        },
    }

    # Display select boxes for hyperparameters and collect selected values
    selected_hyperparams = {}
    hyperparameters = data.columns[6:]
    for param in hyperparameters:
        if param in fixed_hyperparameters.get(model, {}):
            selected_hyperparams[param] = st.sidebar.select_slider(f'Set {param}', fixed_hyperparameters[model][param])


    if st.sidebar.button('Show Results'):
        # Filtering dataset based on selected model and hyperparameters
        mask = (data['Model'] == model)
        for param, value in selected_hyperparams.items():
            mask &= (data[param] == value)

        results = data[mask]

        if not results.empty:
            result = results.iloc[0]
            metrics = {
                "Accuracy": result['Accuracy'],
                "Precision": result['Precision'],
                "Recall": result['Recall'],
                "F1-Score": result['F1-Score'],
                "AUC-ROC": result['AUC-ROC']
            }

            # converting to a table format
            st.write("### Results for the selected configuration:")
            st.table(metrics.items())

            fig, ax = plt.subplots()
            categories = list(metrics.keys())
            values = list(metrics.values())
            ax.bar(categories, values, color='lightblue')
            plt.ylim(0, 1)
            plt.title("Performance Metrics")
            st.pyplot(fig)
        else:
            st.write("boom")