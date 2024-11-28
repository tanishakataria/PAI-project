 # projec members : Emaan : 23i-25260 , Yesaullahsheikh : 23k-0019 , Tanisha : 23k-0067

# Loan Prediction Analysis

## 1. Introduction
The goal of this project is to build and evaluate a logistic regression model to predict loan approval status using the dataset `loan_prediction.csv`. The report outlines the dataset structure, insights derived from analysis, modeling steps, and the evaluation of the results.

### Key Objectives:
1. **Dataset Understanding**: Analyze the structure and attributes of the dataset.
2. **Insights and Patterns**: Uncover trends and relationships within the data.
3. **Model Training**: Train and evaluate a logistic regression model.
4. **Visualization**: Use graphs to enhance understanding and present key insights.

---

## 2. Dataset Overview

### Dataset Description
The dataset contains information about loan applicants, including demographic, financial, and credit details, as well as the loan approval status.

- **Dataset Shape**: The dataset comprises multiple rows and columns.
- **Key Attributes**:
  - **Demographic Information**: `Gender`, `Dependents`, `Education`, `Married` status.
  - **Financial Information**: `Applicant income`, `Coapplicant income`, `Loan amount`, `Loan term`.
  - **Credit Information**: `Credit history`, `Property area`.
  - **Target Variable**: Loan approval status (`Loan_Status`).

### Handling Missing Values
- **Numeric columns** (`LoanAmount`, `Loan_Amount_Term`): Filled with the median value.
- **Categorical columns** (`Gender`, `Married`, `Self_Employed`, `Dependents`): Filled with the mode.

### Preprocessed Data Details:
1. **Scaling**: Numeric features scaled to the range (0,1) using `MinMaxScaler`.
2. **Encoding**: Categorical variables like `Property_Area` were one-hot encoded.
3. **Data Splitting**: The dataset was split into:
   - **Training Set**: 75% of the data.
   - **Validation Set**: 25% of the data.
   - **Test Set**: Provided separately.

---

## 3. Exploratory Data Analysis

### Key Findings:
1. **Applicant Income**:
   - Most applicants have low incomes, with females generally earning less than males.
   - Graduates tend to have higher incomes and request larger loan amounts.
2. **Loan Amount**:
   - Higher loan amounts are generally associated with a positive credit history.
3. **Loan Term**:
   - Standard terms (e.g., 360 months) correlate with higher loan approval rates.
4. **Credit History**:
   - Strongly influences loan approval, with a positive credit history leading to higher approval probabilities.

### Correlation Analysis
- **Credit history** shows the highest positive correlation with loan approval status.
- **Income** and **loan amount** have a weaker, positive correlation.

---

## 4. Visual Insights

### Purpose of Graphs:
Graphs and visualizations are used to:
- Identify trends and patterns.
- Enhance the understanding of relationships between variables.
- Communicate insights effectively.

### Examples of Visualizations:
1. **Histograms and Boxplots**:
   - Illustrate income and loan amount distributions by categories like gender and education.
   - Highlight the impact of outliers on these distributions.
2. **Scatter Plots**:
   - Reveal relationships between applicant income, loan amount, and loan status.
   - Show that positive credit history significantly improves loan approval likelihood.
3. **Heatmaps**:
   - Display correlations between numeric variables, emphasizing the importance of credit history.

---

## 5. Logistic
