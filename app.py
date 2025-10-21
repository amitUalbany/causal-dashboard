# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 00:03:24 2025

@author: Tiger
"""

import streamlit as st
from causal_tool import DataHandler, CausalGraphBuilder, CausalModelManager  # your modules
import pandas as pd


st.title("Causal Inference Dashboard")

# Upload data
uploaded_file = st.file_uploader("Upload CSV Data", type=["csv", "data", "txt"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_ = df.copy()
    st.dataframe(df.head())

    st.sidebar.header("Data Processing Options")
    choice_missing = st.sidebar.selectbox("Handle Missing", ["Drop rows", "Impute mean"])
    choice_confounder = st.sidebar.selectbox("Confounders", ["Education only", "Education + Age"])
    choice_treatment = st.sidebar.selectbox("Treatment", ["Binary Exec/NonExec", "Two Occupations"])
    choice_encoding = st.sidebar.selectbox("Encoding", ["get_dummies", "category_encoders"])

    # Step 1: Data Handling
    handler = DataHandler(df)
    df = handler.handle_missing("1" if choice_missing=="Drop rows" else "2")
    df = handler.select_variables("1" if choice_confounder=="Education only" else "2")
    df = handler.select_treatment_type("1" if choice_treatment=="Binary Exec/NonExec" else "2")
    df = handler.encode_features("1" if choice_encoding=="get_dummies" else "2")
    st.success("Data Encoded Successfully")
    st.dataframe(df.head())
    
    # sidebar for graph options
    st.sidebar.header("Graph Building Options")
    choice_graph = st.sidebar.selectbox("Select Graph option", ["Manual DAG", "DAG using Networkx"]) 
    
    # Step 2: Graph
    graph_builder = CausalGraphBuilder()
    graph_nx, graph_string = graph_builder.define_dag("1" if choice_graph=="Manual DAG" else "2", df) 
    # graph_nx, graph_string = graph_builder.define_dag("2", df)  # choice 1 or 2
    
    # Streamlit visualization
    st.graphviz_chart(graph_string)
    
    # Causal Model
    # Initialize session state variables
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = None
    if 'causal_results' not in st.session_state:
        st.session_state.causal_results = None
    if 'refute_results' not in st.session_state:
        st.session_state.refute_results = None
    if 'shap_fig' not in st.session_state:
        st.session_state.shap_fig = None
    
    # --- Initialize model_manager once ---
    if uploaded_file and st.session_state.model_manager is None:
        treatment = [c for c in df.columns if "occupation" in c][1]
        outcome = [c for c in df.columns if "income" in c][1]
        st.session_state.model_manager = CausalModelManager(df, treatment, outcome, graph_nx)
    
    model_manager = st.session_state.model_manager
    
    # --- Causal Analysis ---
    if st.button("Run Causal Analysis"):
        st.session_state.causal_results = {
            'estimand': model_manager.identify_effect('1'),
            'estimate': model_manager.estimate_effect('1')
        }
    
    # Display causal analysis results if available
    if st.session_state.causal_results:
        st.subheader("Identifying Estimand")
        st.write(st.session_state.causal_results['estimand'])
        st.subheader("Causal Effect Estimation")
        st.write(st.session_state.causal_results['estimate'])
        
    # --- Refutation Analysis ---
    if st.button("Run Refute Analysis"):
        if model_manager.estimand is None or model_manager.estimate is None:
            st.warning("Please run causal analysis first!")
        else:
            st.session_state.refute_results = model_manager.refute_effect()
    
    if st.session_state.refute_results:
        st.subheader("Refutation Analysis")
        r1, r2 = st.session_state.refute_results
        st.write(r1)
        st.write(r2)
    
    # --- SHAP / Explainability ---
    if st.button("Run Explainability Analysis"):
        try:
            import shap
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
    
            # Prepare df
            df_used = pd.DataFrame({
                'education': df_.iloc[:, 3],
                'age': df_.iloc[:, 0],
                'occupation': df_.iloc[:, 6],
                'income': df_.iloc[:, 14]
            })
    
            for col in df_used.select_dtypes(include=['object']).columns:
                df_used[col] = LabelEncoder().fit_transform(df_used[col])
    
            X = df_used[['education', 'age', 'occupation']]
            y = df_used['income']
    
            # Train classifier
            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(X, y)
    
            # Sample for faster SHAP
            X_sample = X.sample(200, random_state=42)
            explainer = shap.TreeExplainer(rf_model, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_sample)
    
            # Handle both binary and multi-class
            if isinstance(shap_values, list):
                shap_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_to_plot = shap_values
            
            # Create SHAP summary plot (do not pre-create figure)
            shap.summary_plot(shap_to_plot, X_sample, plot_type="dot", show=False)
            
            # Grab the current figure from Matplotlib (SHAP drew it)
            fig = plt.gcf()  # get the figure SHAP created
            
            # Render in Streamlit
            # st.pyplot(fig)
            
            # Store for session_state if you want
            st.session_state.shap_fig = fig
            
            # Close to avoid overlap if user clicks multiple times
            plt.close(fig)
            
            
            # Convert SHAP values to array
            shap_arr = np.array(shap_to_plot)
            
            # Handle multi-dimensional SHAP outputs
            if shap_arr.ndim == 3:
                shap_arr = shap_arr.mean(axis=0)  # average over classes
            
            # Handle case where SHAP has fewer features than X (e.g., dropped features)
            n_features_shap = shap_arr.shape[1]
            n_features_X = len(X_sample.columns)
            
            if n_features_shap != n_features_X:
                st.warning(
                    f"Feature count mismatch: SHAP has {n_features_shap}, X has {n_features_X}. "
                    "Attempting to align automatically."
                )
                # Align by trimming or padding
                min_len = min(n_features_shap, n_features_X)
                shap_arr = shap_arr[:, :min_len]
                feature_names = X_sample.columns[:min_len]
            else:
                feature_names = X_sample.columns
            
            # Compute mean absolute SHAP contribution
            feature_contrib = np.abs(shap_arr).mean(axis=0)
            
            # Build contribution DataFrame
            contrib_df = pd.DataFrame({
                "Feature": feature_names,
                "Mean |SHAP| Value": feature_contrib,
                "Contribution (%)": 100 * feature_contrib / feature_contrib.sum()
            }).sort_values(by="Contribution (%)", ascending=False).reset_index(drop=True)
            
            # Display results in Streamlit
            st.markdown("Feature Contribution Summary")
            st.dataframe(contrib_df.style.format({"Contribution (%)": "{:.2f}%"}))
            
            # Optional: plot bar chart for better interpretability
            st.bar_chart(contrib_df.set_index("Feature")["Contribution (%)"])
            
            # Store for session_state
            st.session_state.feature_contrib = contrib_df


        except Exception as e:
            st.warning(f"Explainability module could not run: {e}")
    
    # Display SHAP plot if available
    if st.session_state.shap_fig:
        st.subheader("Feature Importance (SHAP)")
        st.pyplot(st.session_state.shap_fig)
    
    
    # --- Fairness / Bias Analysis ---
    if st.button("Run Bias & Fairness Analysis"):
        st.subheader("Bias and Fairness Analysis")
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            from sklearn.preprocessing import LabelEncoder
            from sklearn.ensemble import RandomForestClassifier
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Prepare df
            df_gender = pd.DataFrame({
                'education': df_.iloc[:, 3],
                'age': df_.iloc[:, 0],
                'occupation': df_.iloc[:, 6],
                'income': df_.iloc[:, 14],
                'gender': df_.iloc[:, 9]
            })
            
            
            # Label Encoder
            le = LabelEncoder()
            for col in df_gender.select_dtypes(include=['object']).columns:
                df_gender[col] = le.fit_transform(df_gender[col])
                
            X_ = df_gender[['education', 'age', 'occupation']]
            y_ = df_gender['income']
            gender = df_gender['gender']
            
            
            # Model training
            model_gender = RandomForestClassifier(random_state=42)
            model_gender.fit(X_, y_)
            y_pred_ = model_gender.predict(X_)
            
            
            # Compute metric by gender
            genders = np.unique(gender)
            metrics = []
            
            for g in genders:
                idx = gender == g
                acc = accuracy_score(y_[idx], y_pred_[idx])
                prec = precision_score(y_[idx], y_pred_[idx], zero_division=0)
                rec = recall_score(y_[idx], y_pred_[idx], zero_division=0)
                metrics.append([g, acc, prec, rec])
                
            metrics_df = pd.DataFrame(metrics, columns=["Gender", "Accuracy", "Precision", "Recall"])
            st.write("Performance by Gender Group:")
            st.dataframe(metrics_df)
            
            # Visualize prediction bia
            y_proba = model_gender.predict_proba(X_)[:,1]
            mean_pred_by_gender = [np.mean(y_proba[gender == g]) for g in genders]

            
            fig, ax = plt.subplots()
            ax.bar(genders, mean_pred_by_gender, color=['skyblue', 'lightcoral'])
            ax.set_xlabel("Gender")
            ax.set_ylabel("Mean predicted probability of High Income")
            ax.set_title("Average Model Prediction by Gender")
            st.pyplot(fig)
            
            # Interpret Bias automatically
            diff = abs(mean_pred_by_gender[0] - mean_pred_by_gender[1])
            if diff > 0.05:
                biased_group = genders[np.argmax(mean_pred_by_gender)]
                st.warning(f"Possible Bias Detected! Model favors **gender {biased_group}**"
                           f"with a {diff:.2f} higher predicted income probability.")
            else:
                st.success("No significant bias detected across gender groups.")
                
            plt.close(fig)
            
        except Exception as e:
            st.error(f"Fairness Analysis Failed: {e}")
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


