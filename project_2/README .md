# Project 2: Regression Challenge

## Executive Summary
---

Property website PropertyGoroUSA, would like to add a Sale Price prediction function for the Ames, Iowa market. Where their agents can have an estimate sale price of the properties listed on their website. To do that we will be looking at what features of a house are most important in predicting its price in the Ames, Iowa market? Would a linear regression model be developed that achieves RMSE error of ~30K?

---

In this project, we will be using the Ames, Iowa Housing Dataset retrieved from [Kaggle](https://www.kaggle.com/c/dsi-us-6-project-2-regression-challenge/data). 
<br>The dataset consist of 2 files, train.csv and test.csv. 
<br>The target variable in this dataset is SalePrice.
<br>Sections below will be how this notebook is organised.

---
```bash
|-- README.md
|-- code
|   |-- project2_Regression.ipynb
|-- data
|   |-- train.csv
|   |-- test.csv
|   |-- data_dictionary.csv
|   |-- heatmap.png
```
---

## Data Dictionary of Ames, Iowa Housing Dataset

| name | description | datatype | vartype |
| :--- | :--- | :--- | :--- |
| pid | Parcel identification number | numeric | Nominal |
| mssubclass | The building class  | categorical | Nominal |
| mszoning | Identifies the general zoning classification of the sale | categorical | Nominal |
| lotfrontage | Linear feet of street connected to property | numeric | Continuous |
| lotarea | Lot size in square feet | numeric | Continuous |
| street | Type of road access to property | categorical | Nominal |
| alley | Type of alley access to property | categorical | Nominal |
| lotshape | General shape of property | categorical | Ordinal |
| landcontour | Flatness of the property | categorical | Nominal |
| utilities | Type of utilities available | categorical | Ordinal |
| lotconfig | Lot configuration | categorical | Nominal |
| landslope | Slope of property | categorical | Ordinal |
| neighborhood | Physical locations within Ames city limits | categorical | Nominal |
| condition1 | Proximity to main road or railroad | categorical | Nominal |
| condition2 | Proximity to main road or railroad (if a second is present) | categorical | Nominal |
| bldgtype | Type of dwelling | categorical | Nominal |
| housestyle | Style of dwelling | categorical | Nominal |
| overallqual | Overall material and finish quality (10:Very Excellent, 1:Very Poor) | categorical | Ordinal |
| overallcond | Overall condition rating (10:Very Excellent, 1:Very Poor) | categorical | Ordinal |
| yearbuilt | Original construction date | numeric | Discrete |
| yearremod/add | Remodel date (same as construction date if no remodeling or additions) | numeric | Discrete |
| roofstyle | Type of roof | categorical | Nominal |
| roofmatl | Roof material | categorical | Nominal |
| exterior1st | Exterior covering on house | categorical | Nominal |
| exterior2nd | Exterior covering on house (if more than one material) | categorical | Nominal |
| masvnrtype | Masonry veneer type | categorical | Nominal |
| masvnrarea | Masonry veneer area in square feet | numeric | Continuous |
| exterqual | Exterior material quality | categorical | Ordinal |
| extercond | Present condition of the material on the exterior | categorical | Ordinal |
| foundation | Type of foundation | categorical | Nominal |
| bsmtqual | Height of the basement (Ex:100+ inches, Po:<70 inches) | categorical | Ordinal |
| bsmtcond | General condition of the basement | categorical | Ordinal |
| bsmtexposure | Walkout or garden level basement walls | categorical | Ordinal |
| bsmtfintype1 | Quality of basement finished area | categorical | Ordinal |
| bsmtfinsf1 | Type 1 finished square feet | numeric | Continuous |
| bsmtfintype2 | Quality of second finished area (if present) | categorical | Ordinal |
| bsmtfinsf2 | Type 2 finished square feet | numeric | Continuous |
| bsmtunfsf | Unfinished square feet of basement area | numeric | Continuous |
| totalbsmtsf | Total square feet of basement area | numeric | Continuous |
| heating | Type of heating | categorical | Nominal |
| heatingqc | Heating quality and condition | categorical | Ordinal |
| centralair | Central air conditioning | categorical | Nominal |
| electrical | Electrical system | categorical | Ordinal |
| 1stflrsf | First Floor square feet | numeric | Continuous |
| 2ndflrsf | Second floor square feet | numeric | Continuous |
| lowqualfinsf | Low quality finished square feet (all floors) | numeric | Continuous |
| grlivarea | Above grade (ground) living area square feet | numeric | Continuous |
| bsmtfullbath | Basement full bathrooms | numeric | Discrete |
| bsmthalfbath | Basement half bathrooms | numeric | Discrete |
| fullbath | Full bathrooms above grade | numeric | Discrete |
| halfbath | Half baths above grade | numeric | Discrete |
| bedroomabvgr | Number of bedrooms above basement level | numeric | Discrete |
| kitchenabvgr | Number of kitchens | numeric | Discrete |
| kitchenqual | Kitchen quality | categorical | Ordinal |
| totrmsabvgrd | Total rooms above grade (does not include bathrooms) | numeric | Discrete |
| functional | Home functionality rating | categorical | Ordinal |
| fireplaces | Number of fireplaces | numeric | Discrete |
| fireplacequ | Fireplace quality | categorical | Ordinal |
| garagetype | Garage location | categorical | Nominal |
| garageyrblt | Year garage was built | numeric | Discrete |
| garagefinish | Interior finish of the garage | categorical | Ordinal |
| garagecars | Size of garage in car capacity | numeric | Discrete |
| garagearea | Size of garage in square feet | numeric | Continuous |
| garagequal | Garage quality | categorical | Ordinal |
| garagecond | Garage condition | categorical | Ordinal |
| paveddrive | Paved driveway | categorical | Ordinal |
| wooddecksf | Wood deck area in square feet | numeric | Continuous |
| openporchsf | Open porch area in square feet | numeric | Continuous |
| enclosedporch | Enclosed porch area in square feet | numeric | Continuous |
| 3ssnporch | Three season porch area in square feet | numeric | Continuous |
| screenporch | Screen porch area in square feet | numeric | Continuous |
| poolarea | Pool area in square feet | numeric | Continuous |
| poolqc | Pool quality | categorical | Ordinal |
| fence | Fence quality | categorical | Ordinal |
| miscfeature | Miscellaneous feature not covered in other categories | categorical | Nominal |
| miscval | $Value of miscellaneous feature | numeric | Continuous |
| mosold | Month Sold | numeric | Discrete |
| yrsold | Year Sold | numeric | Discrete |
| saletype | Type of sale | categorical | Nominal |
| saleprice | Sale price $$ | numeric | Continuous |



