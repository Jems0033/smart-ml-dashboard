import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score,
    f1_score, classification_report
)   

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Model")

# -------------------------
# Cache data (IMPORTANT)
# -------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    st.success("File uploaded successfully!")
    st.write(df)

    columns = df.columns.tolist()

    selected_model = st.selectbox(
        "Select Model:",
        [
            'Linear Regression',
            'Multiple Linear Regression',
            'Polynomial Regression',
            'Random Forest',
            'Decision Tree Classifier',
            'Support Vector Machine',
            'K-Nearest Neighbors'
        ]
    )

    dependent_column = st.selectbox("Select dependent column", columns)

    # -------------------------
    # Feature selection
    # -------------------------
    if selected_model == 'Linear Regression':
        selected_feature = st.selectbox(
            "Select independent column",
            [col for col in columns if col != dependent_column]
        )
        independent_columns = [selected_feature]
    else:
        independent_columns = st.multiselect(
            "Select independent columns",
            [col for col in columns if col != dependent_column]
        )

    # -------------------------
    # Validation
    # -------------------------
    if not independent_columns:
        st.error("Please select at least one independent column")
        st.stop()

    test_size = st.slider("Test size", 0.1, 0.9, 0.2, step=0.05)

    run_model = st.button("Run Model")

    if run_model:

        X = df[independent_columns]
        y = df[dependent_column]

        # Encode categorical features
        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # =========================
        # REGRESSION MODELS
        # =========================
        if selected_model in ['Linear Regression', 'Multiple Linear Regression']:

            if y.dtype == 'object':
                st.error("Regression requires numeric target")
                st.stop()

            model = LinearRegression()
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            st.subheader("📊 Results")

            st.write("Train R²:", r2_score(y_train, train_pred))
            st.write("Test R²:", r2_score(y_test, test_pred))

            st.write("Train MSE:", mean_squared_error(y_train, train_pred))
            st.write("Test MSE:", mean_squared_error(y_test, test_pred))

            st.write("Coefficients:", model.coef_)

            plot_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": test_pred
            })

            st.write("Actual vs Predicted")
            st.scatter_chart(plot_df)

            
            importance = pd.Series(model.coef_, index=X.columns)
            st.write("Feature Importance")
            st.bar_chart(importance.sort_values(ascending=False))

            line_df = pd.DataFrame({
                "Actual": y_test.values,
                "Predicted": test_pred
            }).reset_index(drop=True)

            st.write("Actual vs Predicted (Line Graph)")
            st.line_chart(line_df)
            

        # =========================
        # POLYNOMIAL
        # =========================
        elif selected_model == 'Polynomial Regression':

            if y.dtype == 'object':
                st.error("Polynomial regression requires numeric target")
                st.stop()

            best_r2 = -1
            best_degree = 1

            for i in range(1, 6):  # limited loop
                poly = PolynomialFeatures(degree=i)

                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.transform(X_test)

                temp_model = LinearRegression()
                temp_model.fit(X_train_poly, y_train)

                pred = temp_model.predict(X_test_poly)
                r2 = r2_score(y_test, pred)

                if r2 > best_r2:
                    best_r2 = r2
                    best_degree = i

            st.write("Best degree:", best_degree)

            poly = PolynomialFeatures(degree=best_degree)
            X_train = poly.fit_transform(X_train)
            X_test = poly.transform(X_test)

            model = LinearRegression()
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            st.write("Train R²:", r2_score(y_train, train_pred))
            st.write("Test R²:", r2_score(y_test, test_pred))
            st.write("MSE:", mean_squared_error(y_test, test_pred))

            plot_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": test_pred
            })

            st.write("Actual vs Predicted")
            st.scatter_chart(plot_df)

            feature_names = poly.get_feature_names_out(X.columns)

            importance = pd.Series(model.coef_, index=feature_names)
            st.bar_chart(importance.sort_values(ascending=False).head(10))

            line_df = pd.DataFrame({
                "Actual": y_test.values,
                "Predicted": test_pred
            }).reset_index(drop=True)

            st.write("Actual vs Predicted (Line Graph)")
            st.line_chart(line_df)
        # =========================
        # CLASSIFICATION MODELS
        # =========================
        else:

            # select model
            if selected_model == 'Random Forest':
                best_acc = 0
                best_n = 100

                for n in [10, 50, 100, 200]:
                    rf = RandomForestClassifier(n_estimators=n, random_state=42,max_depth=5)
                    rf.fit(X_train, y_train)

                    pred = rf.predict(X_test)
                    acc = accuracy_score(y_test, pred)

                    if acc > best_acc:
                        best_acc = acc
                        best_n = n

                st.write(f"Best n_estimators: {best_n}")

                model = RandomForestClassifier(n_estimators=best_n, random_state=42)

            elif selected_model == 'Decision Tree Classifier':
                model = DecisionTreeClassifier(criterion="entropy",random_state=42,max_depth=5)

            elif selected_model == 'Support Vector Machine':

                best_acc = 0
                best_C = 1

                for i in range(1, 6):
                    svm = SVC(kernel='rbf',random_state=42,C=i)
                    svm.fit(X_train, y_train)
                    acc = accuracy_score(y_test, svm.predict(X_test))

                    if acc > best_acc:
                        best_acc = acc
                        best_C = i

                model = SVC(kernel='rbf',random_state=42,C=best_C)

            elif selected_model == 'K-Nearest Neighbors':

                best_acc = 0
                best_k = 1

                for i in range(1, 6):
                    knn = KNeighborsClassifier(n_neighbors=i)
                    knn.fit(X_train, y_train)
                    acc = accuracy_score(y_test, knn.predict(X_test))

                    if acc > best_acc:
                        best_acc = acc
                        best_k = i

                model = KNeighborsClassifier(n_neighbors=best_k)

            # train
            model.fit(X_train, y_train)

            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)

            st.subheader("📊 Results")

            st.write("Train Accuracy:", accuracy_score(y_train, pred_train))
            st.write("Test Accuracy:", accuracy_score(y_test, pred_test))

            st.write("Precision:", precision_score(y_test, pred_test, average='weighted', zero_division=0))
            st.write("Recall:", recall_score(y_test, pred_test, average='weighted', zero_division=0))
            st.write("F1 Score:", f1_score(y_test, pred_test, average='weighted', zero_division=0))

            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, pred_test))

            st.write("Classification Report:")
            st.text(classification_report(y_test, pred_test))

            pred_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": pred_test
            })

            st.write("Prediction Distribution")
            st.bar_chart(pred_df["Predicted"].value_counts())

            line_df = pd.DataFrame({
                "Actual": y_test.values,
                "Predicted": pred_test
            }).reset_index(drop=True).head(50)

            st.write("Prediction Trend (First 50 rows)")
            st.line_chart(line_df)

            cm = confusion_matrix(y_test, pred_test)

            # Convert to dataframe
            cm_df = pd.DataFrame(cm)

            # Top 10 rows/cols only
            top_n = 10
            cm_small = cm_df.iloc[:top_n, :top_n]

            st.write("Confusion Matrix (Top 10 classes only)")
            st.dataframe(cm_small)

            compare_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": pred_test
            })
            st.write("Top Predictions vs Actual")
            st.dataframe(compare_df.sample(20))
    
        
