# SAT & ACT participation rate Analysis


### Overview
This project will be looking at SAT and ACT participation rates and scores for both 2017 and 2018 around the United States. The main focus will be on Standardized Testing, Statistical Summaries and Inference.<br><br>
The areas covered by this projects are:
- Basic statistics ()
- Python programming concepts
- Visualizations
- EDA
- Working with Jupyter notebooks for development and reporting

---

### Problem Statement
With the new format for SAT released in March 2016. In this notebook we will be looking at the 2017 and 2018 participation rates by state for both SAT and ACT, to gain insights into which state would benefit most from investments to improve the SAT participation rates.

--- 

### Goals
- Describe data with visualizations & statistical analysis in a Jupyter notebook.
- A non-technical presentation targeting the College board, the organization that administers the SAT. 
- Committing all the deliverables into Github respository.

---

### About the Jupyter Notebook

- Data import and Cleaning
- Description of the data
- Exploratory Data Analysis
- Data Visualization
- Descriptive and Inferential Statisitcs
- Outside research
- Conclusions and Recommendations

---

### Datasets

For this project, two datasets have been provided:
- [2017 SAT Scores](../data/sat_2017.csv)
- [2017 ACT Scores](../data/act_2017.csv)

<br>

#### SAT 2017 Data
| Feature | Type  | Dataset | Description |
|:--|:-:|:-: |:--|
| State | object | SAT | Contains 51 states in America, with no Null values. |
| Participation | int | SAT | Contains 51 participation rate in percentage, with no Null values. Converted participation rates to int, ranging 1 to 100.  |
|   Evidence-Based Reading and Writing    | int | SAT | Contains 51 average score for the Evidence-Based Reading and Writing pert of the test. Values are rounded to whole numbers with no Null values. |
|   Math    | int | SAT | Contains 51 average score for the Math part of the test. Values are rounded to whole numbers with no Null values. |
|   Total    | int | SAT | Contains 51 average combined score for the 2 parts above. Values are rounded to whole numbers with no Null values.  |

#### ACT 2017 Data
| Feature | Type  | Dataset | Description |
|:--|:-:|:-: |:--|
| State | object | ACT | Contains 51 states in America and National (assumation that this is a total average). There are no Null values in this column. |
| Participation | int | ACT | Contains 52 participation rate in percentage, with no Null values. Converted participation rates to int, ranging 1 to 100. |
| English | float | ACT | Contains 52 average score for the English part of the test. Values are rounded to 1 decimal place with no Null values. |
| Math | float | ACT | Contains 52 average score for the Math part of the test. Values are rounded to 1 decimal place with no Null values. |
| Reading | float | ACT | Contains 52 average score for the Reading part of the test. Values are rounded to 1 decimal place with no Null values.  |
| Science | float | ACT | Contains 52 average score for the Science part of the test. Values are rounded to 1 decimal place with no Null values. |
| Composite | float | ACT | Contains 52 average score for the 4 parts above. Values are rounded to 1 decimal place with no Null values. |


