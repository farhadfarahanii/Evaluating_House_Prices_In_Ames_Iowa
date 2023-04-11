# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2: Ames, Iowa Houses sale price

### Overview

My second GA Project covers:
- Basic statistics and probability
- Many Python programming concepts
- Programmatically interacting with files and directories
- Visualizations
- EDA
- Working with Jupyter notebooks for development and reporting

For this project, I'm going to take a look at every aspect of residential homes in Ames, Iowa. I'll seek to identify trends in the data to address my problem statement.

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

### Problem Statement

I work for a home consultation company and aim to provide an evaluation of houses in Ames, Iowa for our customer. Based on my analysis of provided dataset, my focus is on identifying the most effective features on house price and give some recommendations to have a properly investment.

---

### Datasets

In the following you could find my dataset that I used for my project:

* [`final.csv`](./data/final.csv): 

---

### Data Dictionary


|Feature|Type|Dataset|Description|
|---|---|---|---|
|ms_subclass|int|final|The building class|
|lot_frontage|float|final|Linear feet of street connected to property|
|lot_area|int|final|Lot size in square feet|
|overall_qual|int|final|Overall material and finish quality|
|overall_cond|int|final|Overall condition rating|
|year_built|int|final|Original construction date|
|year_remod/add|int|final|Remodel date|
|mas_vnr_area|float|final|Masonry veneer area in square feet|
|bsmtfin_sf_1|float|final|Basement type 1 finished square feet|
|bsmtfin_sf_2|float|final|Basement type 2 finished square feet|
|bsmt_unf_sf|float|final|Unfinished square feet of basement area|
|total_bsmt_sf|float|final|Total square feet of basement area|
|1st_flr_sf|int|final|First Floor square feet|
|2nd_flr_sf|int|final|Second floor square feet|
|low_qual_fin_sf|int|final|Low quality finished square feet (all floors)|
|gr_liv_area|int|final|Above grade (ground) living area square feet|
|bsmt_full_bath|float|final|Basement full bathrooms|
|bsmt_half_bath|float|final|Half baths above grade|
|full_bath|int|final|Full bathrooms above grade|
|half_bath|int|final|Half bathrooms above grade|
|bedroom_abvgr|int|final|Number of bedrooms above grade|
|kitchen_abvgr|int|final|Number of kitchens above grade|
|totrms_abvgrd|int|final|Total rooms above grade (does not include bathrooms)|
|fireplaces|int|final|Number of fireplaces|
|garage_cars|float|final|Size of garage in car capacity|
|garage_area|float|final|Size of garage in square feet|
|wood_deck_sf|int|final|Wood deck area in square feet|
|open porch|int|final|Open porch area in square feet|
|enclosed_porch|int|final|Enclosed porch area in square feet|
|3ssn_porch|int|final|Three season porch area in square feet|
|screen_porch|int|final|Screen porch area in square feet|
|pool_area|int|final|Pool area in square feet|
|misc_val|int|final|Value of miscellaneous feature|
|mo_sold|int|final|Month Sold|
|yr_sold|int|final|Year Sold|
|SalePrice|int|final|the property's sale price in dollars. This is the target variable that you're trying to predict|

---



### My Executive Summary

I did my project with [starter code](./code/project2.ipynb) in a Jupyter notebook. I also have to mention during this project, I got help from https://stackoverflow.com/. In the following I'll describe my methodology and strategies that I used in my project:
 
> 1. Used different pandas methods to get an idea of the rows and columns and types of them.
> 2. Made a new dataset (final) based on numeric columns of train dataset because there were so many unusefull str columns and for modeling I needed only numeric. On my new dataset, made some cleaning, deleting nulls or changing them and also dropped unwanted columns 'garage_yr_blt' and 'pid'.
> 3. First I found correlation between my features and sale price however decided to use all of them to make a stronger model.
> 4. Then by train_test_split made my Xs and ys and standardized them to make a better model.
> 5. my LinearRegression model got a good score on train and test, however used lasso to see if I can get a better number and I did't get that! However lasso helped me to get rid off unusefull columns with 0 coefs. I have to say also I tried get dummies for neighborhood and got a better R^2 score increased by 5% but because I believe my model is already good, and by making some extra columns for data dic, I decided to go in a more simple way! and also I have to say I did't use polynnomial because like get dummies it's only helpfull for making a better score while those extra columns are not realy helpfull in future interpretations and conclusions.
> 6. By making plots through Line Assumptions tried to check Linear Regression, Normality of error and Equality variance of errors in my final model. Afterall I found my model is realy good.
> 6. Tried different plots between target and features with highest coef and correlation to see the relationship between them and making some coclusions. In those plots, found some interesting and shocking relationship like number of parking space or number of bathrooms, different impacts of floors and impacts of other features like fire place and Masonry veneer area.


---


### Conclusions and Recommendations

* For all else held constant, for each 1 sq feet increase in the Ground living area, we expect to increase in sale price by 20,000 dollars and also with the same condition for the first floor we expect to 5,400 dollar increase in sale price.
* Buying a house with more parking spaces could be assumed as good investment for the future, however if even our buyer has enough money but doesn't need spaces for more than 3, it's better to spend that extra money on other parts like spending on increasing ground area.
* My plots clearly shows having more full-bathroom, doesn't necessarily go for increasing the sale price.
* comparing to features like pool, other features like the Masonry veneer area, fireplace, porch and wooden deck are usually ignored while they have a great impact on the sale price, it gets more interesting to know that pool actualy has negative impact on the house price!



