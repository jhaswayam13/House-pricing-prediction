import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=Warning)

dataframe=pd.read_csv(r"E:\Swayam\technical_skills\python\Machine learning\projects\House Pricing prediction\data\train.csv")


pd.set_option('display.max_columns',None)
np.random.seed(13)
sns.set_style('whitegrid')




# plotting the distibution of target variable i..e SalePrice
data=dataframe['SalePrice']

sns.boxplot(data=data,color='skyblue')
pl.axhline(data.median(), color='red', linestyle='dashed', linewidth=1.5, label='Median') # adding median line

pl.axhline(data.quantile(0.25), color='green', linestyle='dashed', linewidth=1, label='25th Percentile') # adding 25 %ile line
pl.axhline(data.quantile(0.75), color='green', linestyle='dashed', linewidth=1, label='75th Percentile') # adding 75% ile line

pl.ylabel('Sale Price')
pl.title('Box Plot of Sale Price')

pl.legend()

pl.show()




# splitted 75-25(%) to train and val data further diving into input and targetted dataframes/pandas.core series 


val_data,train_data=train_test_split(dataframe,test_size=0.75,random_state=13)

input_cols=list(train_data.columns[1:-1])

cols_to_unalive=['YearBuilt','YearRemodAdd','GarageYrBlt']

for cols in cols_to_unalive:
    if cols in input_cols:
        input_cols.remove(cols)

target_cols='SalePrice'


train_input=train_data[input_cols]
train_target=train_data[target_cols]
val_input=val_data[input_cols]
val_target=val_data[target_cols]

# separating numerical and categorical columns

numerical_cols=train_input.select_dtypes(include=np.number).columns.tolist()
categorical_cols=train_input.select_dtypes(include='object').columns.tolist()

false_numerical=['MSSubClass']

for col in false_numerical:
    if col in numerical_cols:
        numerical_cols.remove(col)
        categorical_cols.append(col)

numerical_cols.remove('YrSold')

# let us now fill the nan values in mumerical cols with average value of thatv column using simple impoter

imputer=SimpleImputer(strategy='mean')

imputer.fit(dataframe[numerical_cols])

train_input[numerical_cols]=imputer.transform(train_input[numerical_cols])
val_input[numerical_cols]=imputer.transform(val_input[numerical_cols])

# and now try to create a ordinary encoding label manually

ordinal_cols_t1=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','FireplaceQu','GarageFinish','GarageQual','GarageCond','PoolQC','Fence']
ordinal_cols_t2=['ExterCond','ExterQual','KitchenQual']     # ordinal encoding without NA/0
ordinal_cols_t3=['Functional']
ordinal_cols_t4='YrSold'





# creating an ordinal labelling for t1 and t2 t3 and t4 manually
dict_t1={

    'Ex':5,
    'Gd':4,
    'TA':3,
    'Fa':2,
    'Po':1,
    'NA':0
}

dict_t2={

    'Ex':5,
    'Gd':4,
    'TA':3,
    'Fa':2,
    'Po':1

}

dict_t3={

    'Min1':7,
    'Min2':6,
    'Mod':5,
    'Maj1':4,
    'Maj2':3,
    'Sev':2,
    'Sal':1

}

dict_t4={

    2006:1,
    2007:2,
    2008:3,
    2009:4,
    2010:5



}

for col1 in ordinal_cols_t1:
    train_input[col1]=train_input[col1].map(dict_t1)

for col2 in ordinal_cols_t2:
    train_input[col2]=train_input[col2].map(dict_t2)

train_input['Functional']=train_input['Functional'].map(dict_t3)

train_input['YrSold']=train_input['YrSold'].map(dict_t4)

for col1 in ordinal_cols_t1:
    val_input[col1]=val_input[col1].map(dict_t1)

for col2 in ordinal_cols_t2:
    val_input[col2]=val_input[col2].map(dict_t2)

val_input['Functional']=val_input['Functional'].map(dict_t3)
val_input['YrSold']=val_input['YrSold'].map(dict_t4)



for cols in ordinal_cols_t1 + ordinal_cols_t2+ordinal_cols_t3:
    if cols in categorical_cols:
        categorical_cols.remove(cols)
        numerical_cols.append(cols)

numerical_cols.append(ordinal_cols_t4)

too_many_na=['Functional','Fence','PoolQC','GarageFinish','BsmtFinType2','BsmtFinType1','BsmtExposure']

train_input.drop(columns=too_many_na,inplace=True)
val_input.drop(columns=too_many_na,inplace=True)

for cols in too_many_na:
    if cols in numerical_cols:
        numerical_cols.remove(cols)

print(numerical_cols)
print(type(numerical_cols))
train_input[numerical_cols]=train_input[numerical_cols].fillna(0)
val_input[numerical_cols]=val_input[numerical_cols].fillna(0)

# scaling down the numerical columns to (0,1) using MINMAXSCALER

'''scaler=MinMaxScaler()

scaler.fit(dataframe[numerical_cols])

train_input[numerical_cols]=scaler.transform(train_input[numerical_cols])
val_input[numerical_cols]=scaler.transform(val_input[numerical_cols])'''


# using one hot encoding for categorical columns



encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
encoder.fit(train_input[categorical_cols])

encoded_cols=list(encoder.get_feature_names_out(categorical_cols))
train_input[encoded_cols]=encoder.transform(train_input[categorical_cols])
val_input[encoded_cols]=encoder.transform(val_input[categorical_cols])


#Now finally for the climax i.e simple linear regression model aka least square method

linear=LinearRegression()
input_for_model=train_input[numerical_cols + encoded_cols]





linear.fit(input_for_model,train_target)
predictions_train=linear.predict(input_for_model)



featured_weights=pd.DataFrame({
    'Features': (numerical_cols+encoded_cols),
    'Weights': linear.coef_



})



initial_feature=featured_weights.iloc[0:21]


# create a bargraph about weights for different features
X=initial_feature['Features']
Y=initial_feature['Weights']

pl.figure(figsize=(20,10))
ax=sns.barplot(x=X,y=Y, data=initial_feature)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
pl.show()

# now let's evaluate how accurate was our model for training dataset

 


mae=mean_absolute_error(train_target,predictions_train)
mse=mean_squared_error(train_target,predictions_train)
r2=r2_score(train_target,predictions_train)

print("This one is for training dataset")
print(f"{mae} is the mean absolute error\n {mse} is mean squared error \n and {r2} is the R2 score for the linear regression  model")

metrics_for_train=np.array([mae,mse,r2])
metrics_for_train = metrics_for_train.reshape(1, -1)




#Now test how our model perform for vlaidation dataset
val_input_for_model=val_input[numerical_cols + encoded_cols]

predictions_val=linear.predict(val_input_for_model)
pl.subplot(1,2,1)
sns.scatterplot(x=train_target,y=predictions_train)
pl.title("Relation between prediction and actual target for training")

pl.subplot(1,2,2)

sns.scatterplot(x=val_target,y=predictions_val)
pl.title("Relation between prediction and actual target for validation")
pl.show()



mae_val=mean_absolute_error(val_target,predictions_val)
mse_val=mean_squared_error(val_target,predictions_val)
r2_val=r2_score(val_target,predictions_val)


print("This one is for training dataset")
print(f"{mae_val} is the mean absolute error\n {mse_val} is mean squared error \n and {r2_val} is the R2 score for the linear regression  model")

metrics_for_val=np.array([mae_val,mse_val,r2_val])

metrics_for_val = metrics_for_val.reshape(1, -1)



# plotting heatmap for demonstrating metrics of our model for both training and validation dataset

pl.subplot(1,2,1)
sns.heatmap(metrics_for_train,annot=True,xticklabels=['MAE', 'MSE', 'R2'])
pl.xlabel('Metrics')
pl.title('Performance Metrics for Training Dataset')

pl.subplot(1,2,2)

sns.heatmap(metrics_for_val,annot=True,xticklabels=['MAE', 'MSE', 'R2'])
pl.xlabel('Metrics')
pl.title('Performance Metrics for Validation Dataset')




pl.show()





