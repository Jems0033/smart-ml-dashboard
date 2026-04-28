import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Maximum number of unique target values to treat a column as classification.
# Beyond this threshold, the confusion matrix becomes unreadable and the
# task is better treated as regression.
_MAX_CLASSES_FOR_CONFUSION_MATRIX = 20

st.set_page_config(page_title="Smart ML Dashboard", layout="wide")
st.title("📊 Smart ML Dashboard")


# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)


def split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ── regression visualisations ─────────────────────────────────────────────

def plot_regression(y_test, y_pred, title="Regression Results"):
    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6, color="steelblue", edgecolors="k", linewidths=0.4)
        mn, mx = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Ideal")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots()
        idx = np.arange(len(y_test))
        ax.plot(idx, np.array(y_test), label="Actual", color="steelblue")
        ax.plot(idx, y_pred, label="Predicted", color="orange", linestyle="--")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        ax.set_title("Prediction Trend")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col3:
        st.write("*Third panel: see Feature Importance below.*")


def plot_feature_importance_regression(model, feature_names):
    """Works for LinearRegression / Pipeline ending in LinearRegression."""
    try:
        if hasattr(model, "named_steps"):
            lr = model.named_steps.get("linearregression")
            coefs = lr.coef_ if lr is not None else None
        else:
            coefs = model.coef_
        if coefs is None:
            return
        coefs = np.atleast_1d(coefs)
        # For polynomial regression the transformed feature count differs from
        # the original feature names so skip if lengths don't match.
        if len(coefs) != len(feature_names):
            st.info("Feature importance not shown for polynomial features.")
            return
        fig, ax = plt.subplots(figsize=(6, max(3, len(feature_names) * 0.4)))
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, coefs, color="steelblue", edgecolor="k", linewidth=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Coefficient")
        ax.set_title("Feature Importance (Coefficients)")
        st.pyplot(fig)
        plt.close(fig)
    except (AttributeError, ValueError, TypeError) as exc:
        st.warning(f"Feature importance unavailable: {exc}")


# ── classification visualisations ────────────────────────────────────────

def plot_classification(y_test, y_pred):
    classes = np.unique(np.concatenate([y_test, y_pred]))
    n_classes = len(classes)

    col1, col2 = st.columns(2)

    with col1:
        if n_classes <= _MAX_CLASSES_FOR_CONFUSION_MATRIX:
            cm = confusion_matrix(y_test, y_pred, labels=classes)
            fig, ax = plt.subplots(figsize=(max(4, n_classes), max(3, n_classes - 1)))
            sns.heatmap(cm, annot=(n_classes <= 15), fmt="d", cmap="Blues",
                        xticklabels=classes, yticklabels=classes, ax=ax,
                        linewidths=0.5, linecolor="lightgray")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(f"Confusion matrix skipped ({n_classes} classes – too many to display meaningfully).")
    with col2:
        unique, counts = np.unique(y_pred, return_counts=True)
        fig, ax = plt.subplots()
        ax.bar([str(u) for u in unique], counts, color="steelblue",
               edgecolor="k", linewidth=0.4)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Count")
        ax.set_title("Prediction Distribution")
        st.pyplot(fig)
        plt.close(fig)

    col3, col4 = st.columns(2)
    with col3:
        n_show = min(100, len(y_test))
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(np.array(y_test)[:n_show], label="Actual", color="steelblue")
        ax.plot(np.array(y_pred)[:n_show], label="Predicted",
                color="orange", linestyle="--")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Class")
        ax.set_title("Prediction Trend (first 100 samples)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        n_show = min(30, len(y_test))
        comparison = pd.DataFrame({
            "Sample": range(n_show),
            "Actual": np.array(y_test)[:n_show],
            "Predicted": np.array(y_pred)[:n_show],
        })
        fig, ax = plt.subplots(figsize=(10, 3))
        x = np.arange(n_show)
        ax.scatter(x, comparison["Actual"], label="Actual", color="steelblue",
                   marker="o", s=30)
        ax.scatter(x, comparison["Predicted"], label="Predicted",
                   color="orange", marker="x", s=30)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Class")
        ax.set_title("Sample Comparison (first 30)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)


# ── auto-tune helpers ─────────────────────────────────────────────────────

def best_poly_degree(X_train, y_train, X_test, y_test, max_degree=6):
    best_deg, best_r2 = 1, -np.inf
    for d in range(1, max_degree + 1):
        model = make_pipeline(PolynomialFeatures(d), LinearRegression())
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        if r2 > best_r2:
            best_r2, best_deg = r2, d
    return best_deg


def best_n_estimators(X_train, y_train, X_test, y_test, options=(50, 100, 150, 200)):
    best_n, best_acc = options[0], -np.inf
    for n in options:
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if acc > best_acc:
            best_acc, best_n = acc, n
    return best_n


def best_svm_C(X_train, y_train, X_test, y_test, options=(0.1, 1.0, 10.0)):
    best_c, best_acc = options[0], -np.inf
    for c in options:
        model = SVC(C=c, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if acc > best_acc:
            best_acc, best_c = acc, c
    return best_c


def best_knn_k(X_train, y_train, X_test, y_test, max_k=15):
    best_k, best_acc = 3, -np.inf
    for k in range(1, max_k + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if acc > best_acc:
            best_acc, best_k = acc, k
    return best_k


# ── sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

# ── main content ──────────────────────────────────────────────────────────

if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
    st.stop()

df = load_data(uploaded_file)

st.subheader("📁 Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)
st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

# Drop non-numeric columns for feature selection (but keep all for display)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
excluded = [c for c in df.columns if c not in numeric_cols]
if excluded:
    st.info(
        f"ℹ️ {len(excluded)} non-numeric column(s) excluded from feature/target selection: "
        + ", ".join(f"`{c}`" for c in excluded)
    )
if len(numeric_cols) < 2:
    st.error("The dataset must contain at least 2 numeric columns.")
    st.stop()

st.subheader("🎛️ Feature & Target Selection")
col_a, col_b = st.columns(2)
with col_a:
    target_col = st.selectbox("Target Column", options=numeric_cols)
with col_b:
    feature_cols = st.multiselect(
        "Feature Columns",
        options=[c for c in numeric_cols if c != target_col],
        default=[c for c in numeric_cols if c != target_col],
    )

if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

X = df[feature_cols].dropna()
y = df.loc[X.index, target_col]

# Remove rows where target is NaN
mask = y.notna()
X, y = X[mask], y[mask]

if len(X) < 10:
    st.error("Not enough samples after dropping NaN rows (need at least 10).")
    st.stop()

# ── task type ─────────────────────────────────────────────────────────────

n_unique = y.nunique()
inferred_task = "Classification" if n_unique <= 20 else "Regression"

st.subheader("🤖 Model Selection")
task = st.radio("Task Type", ["Regression", "Classification"],
                index=0 if inferred_task == "Regression" else 1,
                horizontal=True)

if task == "Regression":
    model_name = st.selectbox(
        "Regression Model",
        ["Linear Regression (single feature)",
         "Multiple Linear Regression (multiple features)",
         "Polynomial Regression (auto degree)"],
    )
else:
    model_name = st.selectbox(
        "Classification Model",
        ["Random Forest (auto n_estimators)", "Decision Tree",
         "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)"],
    )

test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)

run_button = st.button("▶️ Train & Evaluate")

if not run_button:
    st.stop()

# ── train/test split ──────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = split(X, y, test_size=test_size)

# ── regression pipeline ───────────────────────────────────────────────────

if task == "Regression":
    st.subheader("📈 Regression Results")

    if model_name in ("Linear Regression (single feature)",
                      "Multiple Linear Regression (multiple features)"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        label = model_name
    else:
        with st.spinner("Auto-selecting polynomial degree…"):
            deg = best_poly_degree(X_train, y_train, X_test, y_test)
        st.info(f"Best polynomial degree selected: **{deg}**")
        model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
        model.fit(X_train, y_train)
        label = f"Polynomial Regression (degree={deg})"

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)

    m1, m2, m3 = st.columns(3)
    m1.metric("R² (Train)", f"{r2_train:.4f}")
    m2.metric("R² (Test)", f"{r2_test:.4f}")
    m3.metric("MSE (Test)", f"{mse:.4f}")

    st.markdown(f"**Model:** {label}")
    plot_regression(y_test, y_pred_test)

    st.subheader("📊 Feature Importance")
    plot_feature_importance_regression(model, feature_cols)

# ── classification pipeline ───────────────────────────────────────────────

else:
    st.subheader("📈 Classification Results")
    # Encode target labels to ensure consistent, well-ordered class labels
    # (avoids lexicographic surprises with numeric class values like 1, 2, 10).
    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_test]))
    y_train_cls = pd.Series(le.transform(y_train), index=y_train.index)
    y_test_cls = pd.Series(le.transform(y_test), index=y_test.index)

    if model_name == "Random Forest (auto n_estimators)":
        with st.spinner("Auto-selecting n_estimators…"):
            n_est = best_n_estimators(X_train, y_train_cls, X_test, y_test_cls)
        st.info(f"Best n_estimators selected: **{n_est}**")
        model = RandomForestClassifier(n_estimators=n_est, random_state=42)

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)

    elif model_name == "Support Vector Machine (SVM)":
        with st.spinner("Auto-selecting SVM C value…"):
            c_val = best_svm_C(X_train, y_train_cls, X_test, y_test_cls)
        st.info(f"Best SVM C value selected: **{c_val}**")
        model = SVC(C=c_val, random_state=42)

    else:  # KNN
        with st.spinner("Auto-selecting KNN neighbors…"):
            k_val = best_knn_k(X_train, y_train_cls, X_test, y_test_cls)
        st.info(f"Best KNN k selected: **{k_val}**")
        model = KNeighborsClassifier(n_neighbors=k_val)

    model.fit(X_train, y_train_cls)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    avg = "binary" if y_train_cls.nunique() == 2 else "weighted"

    acc_train = accuracy_score(y_train_cls, y_pred_train)
    acc_test = accuracy_score(y_test_cls, y_pred_test)
    prec = precision_score(y_test_cls, y_pred_test, average=avg, zero_division=0)
    rec = recall_score(y_test_cls, y_pred_test, average=avg, zero_division=0)
    f1 = f1_score(y_test_cls, y_pred_test, average=avg, zero_division=0)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy (Train)", f"{acc_train:.4f}")
    m2.metric("Accuracy (Test)", f"{acc_test:.4f}")
    m3.metric("Precision", f"{prec:.4f}")
    m4.metric("Recall", f"{rec:.4f}")
    m5.metric("F1 Score", f"{f1:.4f}")

    with st.expander("📄 Classification Report"):
        st.text(classification_report(y_test_cls, y_pred_test, zero_division=0))

    plot_classification(y_test_cls, y_pred_test)
