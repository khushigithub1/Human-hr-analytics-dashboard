import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


# LOAD MACHINE LEARNING MODEL
model = pickle.load(
    open("employee_performance_model.pkl", "rb")
)


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_excel("Future_INX.xls", engine="xlrd")

# =====================================================
# LOAD MACHINE LEARNING MODEL
# =====================================================

model = pickle.load(open("model.pkl", "rb"))

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

.stMetric {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER SECTION
# =====================================================

st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>
📊 Human HR Analytics Dashboard
</h1>

<h4 style='text-align: center; color: white;'>
Employee Performance & Workforce Insights
</h4>
""", unsafe_allow_html=True)

st.info("""
This dashboard helps HR teams analyze workforce trends,
employee distribution, attrition patterns,
and employee satisfaction metrics interactively.
""")

st.markdown("---")

# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.header("🔍 Filters")

st.sidebar.info("""
Analyze workforce trends,
employee distribution,
and attrition insights.
""")

# Department Filter

department_options = ["All"] + list(df["EmpDepartment"].unique())

selected_department = st.sidebar.selectbox(
    "Select Department",
    department_options
)

# Gender Filter

gender_options = ["All"] + list(df["Gender"].unique())

selected_gender = st.sidebar.selectbox(
    "Select Gender",
    gender_options
)

# =====================================================
# FILTER DATA
# =====================================================

filtered_df = df.copy()

if selected_department != "All":
    filtered_df = filtered_df[
        filtered_df["EmpDepartment"] == selected_department
    ]

if selected_gender != "All":
    filtered_df = filtered_df[
        filtered_df["Gender"] == selected_gender
    ]

# =====================================================
# EMPTY DATA CHECK
# =====================================================

if filtered_df.empty:
    st.warning("No data available for selected filters")
    st.stop()

# =====================================================
# KPI CARDS
# =====================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "👨 Total Employees",
    len(filtered_df)
)

col2.metric(
    "🏢 Departments",
    filtered_df["EmpDepartment"].nunique()
)

col3.metric(
    "🎂 Average Age",
    round(filtered_df["Age"].mean(), 1)
)

col4.metric(
    "⭐ Avg Satisfaction Score",
    round(filtered_df["EmpJobSatisfaction"].mean(), 1)
)

st.caption(
    f"Showing {filtered_df.shape[0]} rows and {filtered_df.shape[1]} columns"
)

st.markdown("---")

# =====================================================
# DATASET TABLE SECTION
# =====================================================

st.subheader("📄 Employee Dataset Preview")

with st.expander("View Complete Dataset"):
    st.dataframe(filtered_df)

# DOWNLOAD BUTTON

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="⬇ Download Filtered Dataset",
    data=csv,
    file_name="HR_Analytics.csv",
    mime="text/csv"
)

st.markdown("---")

# =====================================================
# CHARTS SECTION
# =====================================================

st.subheader("📊 HR Analytics Visualizations")

# =====================================================
# FIRST ROW
# =====================================================

col1, col2 = st.columns(2)

# GENDER PIE CHART

fig1 = px.pie(
    filtered_df,
    names="Gender",
    title="Gender Distribution",
    hole=0.4
)

# DEPARTMENT BAR CHART

dept_count = filtered_df["EmpDepartment"]\
    .value_counts()\
    .sort_values(ascending=False)

fig2 = px.bar(
    x=dept_count.index,
    y=dept_count.values,
    labels={
        "x": "Department",
        "y": "Employees"
    },
    title="Department Distribution",
    text=dept_count.values
)

fig2.update_traces(marker_color='#4CAF50')

with col1:
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# SECOND ROW
# =====================================================

col3, col4 = st.columns(2)

# ATTRITION ANALYSIS

fig3 = px.histogram(
    filtered_df,
    x="Attrition",
    title="Attrition Analysis",
    text_auto=True
)

# AGE DISTRIBUTION

fig4 = px.histogram(
    filtered_df,
    x="Age",
    nbins=20,
    title="Employee Age Distribution"
)

with col3:
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.plotly_chart(fig4, use_container_width=True)

# =====================================================
# THIRD ROW
# =====================================================

st.subheader("😊 Job Satisfaction Analysis")

fig5 = px.box(
    filtered_df,
    y="EmpJobSatisfaction",
    color="Gender",
    title="Job Satisfaction Distribution"
)

st.plotly_chart(fig5, use_container_width=True)

# =====================================================
# CORRELATION HEATMAP
# =====================================================

st.subheader("🔥 Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include='number')

corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(12, 6))

sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=False
)

st.pyplot(fig)

st.markdown("---")

# =====================================================
# BUSINESS INSIGHTS SECTION
# =====================================================

st.subheader("📌 Business Insights & Recommendations")

st.info("""
• Salary hike percentage was identified as the most influential factor affecting employee performance.

• Employees with low salary growth showed lower performance ratings and engagement levels.

• Environment satisfaction emerged as one of the strongest contributors to employee productivity and retention.

• Employees staying in the same role for extended periods without promotion demonstrated reduced performance trends.

• The predictive machine learning model achieved approximately 92.9% accuracy in classifying employee performance categories.
""")

# =====================================================
# FINAL RECOMMENDATIONS
# =====================================================

st.subheader("✅ Strategic HR Recommendations")

st.success("""
• Implement a transparent and structured salary review process to reward high-performing employees consistently.

• Conduct regular employee satisfaction surveys to monitor workplace environment and team engagement.

• Introduce role rotation programs and timely promotion opportunities to reduce employee stagnation.

• Focus on improving workplace culture and managerial support in departments with lower satisfaction levels.

• Utilize the predictive performance model as a decision-support tool during appraisals and workforce planning.

• Identify at-risk employees early and provide proactive interventions to improve long-term retention and productivity.
""")

st.markdown("---")

# =====================================================
# EMPLOYEE PERFORMANCE PREDICTION
# =====================================================

st.subheader("🤖 Employee Performance Prediction")

age = st.number_input(
    "Employee Age",
    min_value=18,
    max_value=60
)

distance = st.number_input(
    "Distance From Home",
    min_value=1,
    max_value=50
)

satisfaction = st.slider(
    "Job Satisfaction",
    1,
    5
)

if st.button("Predict Performance"):

    prediction = model.predict([[
        age,
        distance,
        satisfaction
    ]])

    st.success(
        f"Predicted Performance Rating: {prediction[0]}"
    )

st.markdown("---")

# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div style='text-align: center;'>

### 🚀 HR Analytics Dashboard Project

Created using Python, Pandas, Plotly, Seaborn, Machine Learning & Streamlit

</div>
""", unsafe_allow_html=True)