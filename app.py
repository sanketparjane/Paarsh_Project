import streamlit as st
import pandas as pd
import pickle

# -------------------- Load Files --------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------- Title --------------------
st.set_page_config(page_title="Employee Attrition Prediction")
st.title("Employee Attrition Prediction")
st.write("Enter employee details below:")

# -------------------- Inputs --------------------
satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.50)

last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.50)

number_project = st.number_input(
    "Number of Projects",
    min_value=1,
    max_value=10,
    value=3
)

average_monthly_hours = st.number_input(
    "Average Monthly Hours",
    min_value=50,
    max_value=350,
    value=200
)

time_spend_company = st.number_input(
    "Years at Company",
    min_value=1,
    max_value=20,
    value=3
)

work_accident = st.selectbox(
    "Work Accident",
    [0, 1]
)

promotion_last_5years = st.selectbox(
    "Promotion in Last 5 Years",
    [0, 1]
)

salary = st.selectbox(
    "Salary",
    ["low", "medium", "high"]
)

dept = st.selectbox(
    "Department",
    [
        "IT",
        "RandD",
        "accounting",
        "hr",
        "management",
        "marketing",
        "product_mng",
        "sales",
        "support",
        "technical"
    ]
)

# -------------------- Prediction --------------------
if st.button("Predict"):

    # Numerical Data
    num_df = pd.DataFrame({
        "satisfactoryLevel": [satisfaction_level],
        "lastEvaluation": [last_evaluation],
        "numberOfProjects": [number_project],
        "avgMonthlyHours": [average_monthly_hours],
        "timeSpent.company": [time_spend_company],
        "workAccident": [work_accident],
        "promotionInLast5years": [promotion_last_5years]
    })

    # Categorical Data
    cat_df = pd.DataFrame({
        "salary": [salary],
        "dept": [dept]
    })

    # Encode
    encoded = encoder.transform(cat_df)

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out()
    )

    # Combine
    final_data = pd.concat([num_df, encoded_df], axis=1)

    # Match model column order
    final_data = final_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Scale
    final_scaled = scaler.transform(final_data)

    # Predict
    prediction = model.predict(final_scaled)[0]

    probability = model.predict_proba(final_scaled)[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.error("⚠️ Employee is likely to leave the company.")
    else:
        st.success("✅ Employee is likely to stay in the company.")

    st.write(f"Probability of Leaving: {probability:.2%}")
