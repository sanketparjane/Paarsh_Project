import streamlit as st
import pandas as pd
import pickle

# Load saved files
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Employee Attrition Prediction")

st.write("Enter employee details:")

# User Inputs
satisfaction_level = st.number_input("Satisfaction Level", 0.0, 1.0, 0.5)
last_evaluation = st.number_input("Last Evaluation", 0.0, 1.0, 0.5)
number_project = st.number_input("Number of Projects", 1, 20, 3)
average_montly_hours = st.number_input("Average Monthly Hours", 50, 400, 200)
time_spend_company = st.number_input("Years at Company", 1, 20, 3)
Work_accident = st.selectbox("Work Accident", [0, 1])
promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])
salary = st.selectbox("Salary", ["low", "medium", "high"])
dept = st.selectbox(
    "Department",
    ["IT", "RandD", "accounting", "hr", "management",
     "marketing", "product_mng", "sales",
     "support", "technical"]
)

if st.button("Predict"):

    # Encode categorical columns
    cat_df = pd.DataFrame({
        "salary": [salary],
        "dept": [dept]
    })

    encoded = encoder.transform(cat_df)
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(["salary", "dept"])
    )

    # Numerical data
    num_df = pd.DataFrame({
        "satisfaction_level": [satisfaction_level],
        "last_evaluation": [last_evaluation],
        "number_project": [number_project],
        "average_montly_hours": [average_montly_hours],
        "time_spend_company": [time_spend_company],
        "Work_accident": [Work_accident],
        "promotion_last_5years": [promotion_last_5years]
    })

    # Scale numerical columns
    num_df[num_df.columns] = scaler.transform(num_df)

    # Final input
    final_data = pd.concat([num_df, encoded_df], axis=1)

    # Prediction
    prediction = model.predict(final_data)[0]
    probability = model.predict_proba(final_data)[0][1]

    if prediction == 1:
        st.error("Employee is likely to leave the company.")
    else:
        st.success("Employee is likely to stay in the company.")

    st.write(f"Probability of Leaving: **{probability:.2%}**")