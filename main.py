import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score


warnings.filterwarnings("ignore", category=Warning)


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows', None)
np.random.seed(13)
sns.set_style('whitegrid')



class Reading_csv:
    def __init__(self,filepath):
        self.filepath=filepath

    def read(self):
        try:
            self.dataframe=pd.read_csv(self.filepath)
            return self

        except FileNotFoundError:
            self.dataframe=0
            return self

path=r"E:\Swayam\technical_skills\python\OOPs\datad\data_HP\train.csv"

Reader=Reading_csv(path)
dataframe=Reader.read().dataframe   # dataframe created 



sns.boxplot(data=dataframe,color='skyblue')
pl.axhline(dataframe.median(), color='red', linestyle='dashed', linewidth=1.5, label='Median') # adding median line

pl.axhline(dataframe.quantile(0.25), color='green', linestyle='dashed', linewidth=1, label='25th Percentile') # adding 25 %ile line
pl.axhline(dataframe.quantile(0.75), color='green', linestyle='dashed', linewidth=1, label='75th Percentile') # adding 75% ile line

pl.ylabel('Sale Price')
pl.title('Box Plot of Sale Price')

pl.legend()

pl.show()

class splitting_dataset(Reading_csv):
    def __init__(self,filepath):
        super().__init__(filepath)
        self.read()

    def split(self):
        self.val_data,self.train_data=train_test_split(self.dataframe,test_size=0.75,random_state=13)
        return self
        

train_data=splitting_dataset(path).split().train_data
val_data=splitting_dataset(path).split().val_data

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







class column_identification(Reading_csv):
    def __init__(self,filepath):
        super().__init__(filepath)
        self.read()

    def numerical_categorical(self,data_to_change):
        self.num_col=data_to_change.select_dtypes(include=np.number).columns.to_list()
        self.cat_col=data_to_change.select_dtypes(include='object').columns.to_list()
        return self


numerical_col=column_identification(path).numerical_categorical(train_input).num_col
catgeorical_col=column_identification(path).numerical_categorical(train_input).cat_col
print(numerical_col)


class feature_engineering(Reading_csv):
    def __init__(self,filepath):
        super().__init__(filepath)
        self.read()

    def col_appender(self,col_to_append,col_list):
        col_to_append.extend(col_list)

    def col_remover(self,col_to_remove,col_list):
        for elements in col_list:
            col_to_remove.remove(elements)
    def null_remover(self,whom_to_remove,col_list):
        whom_to_remove.drop(columns=col_list,inplace=True)
        

false_num=['MSSubClass']


feature_engineering(path).col_remover(numerical_col,false_num)      # MSSubClass has been removed from numerical_col and added to categorical col
feature_engineering(path).col_appender(catgeorical_col,false_num)

numerical_col.remove('YrSold')
catgeorical_col.append('YrSold')
        

class simple_imputer(feature_engineering):
    def __init__(self,path):
        super().__init__(path)

    def imputing(self,fitter,who_fit,cols_to_fit):
        imputer=SimpleImputer(strategy='mean')
        imputer.fit(fitter[cols_to_fit])
        who_fit[cols_to_fit]=imputer.transform(who_fit[cols_to_fit])
        
imputer=simple_imputer(path)


imputer.imputing(dataframe,train_input,numerical_col)
imputer.imputing(dataframe,val_input,numerical_col)





class Encoding(feature_engineering):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')

    def label_encoder(self,whom_to_label,col_list,dictionary_to_map):
        for col in col_list:
            whom_to_label[col]=whom_to_label[col].map(dictionary_to_map)

    def one_hot_encoder(self,whom_to_encode,col_list):
       
        self.encoder.fit(whom_to_encode[col_list])
        return self.encoder
        
    def encoding_stage2(self,whom_to_encode,col_list):
 
        encoded_cols=list(self.encoder.get_feature_names_out(col_list))
        whom_to_encode[encoded_cols]=self.encoder.transform(whom_to_encode[catgeorical_col])
        self.encoded_cols=encoded_cols
        return encoded_cols






dict_1= {

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
        

t1_col_to_encode=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','FireplaceQu','GarageFinish','GarageQual','GarageCond','PoolQC','Fence']
t2_col_to_encode=['ExterCond','ExterQual','KitchenQual']
t3_col_to_encode=['Functional']


Encoder=Encoding(path)
# encoding training data
Encoder.label_encoder(train_input,t1_col_to_encode,dict_1)
Encoder.label_encoder(train_input,t2_col_to_encode,dict_1)
Encoder.label_encoder(train_input,t3_col_to_encode,dict_1)


# encoding validation data
Encoder.label_encoder(val_input,t1_col_to_encode,dict_1)
Encoder.label_encoder(val_input,t2_col_to_encode,dict_1)
Encoder.label_encoder(val_input,t3_col_to_encode,dict_1)





Encoder.one_hot_encoder(train_input,catgeorical_col)
encoded_cols=Encoder.encoding_stage2(train_input,catgeorical_col)
Encoder.encoding_stage2(val_input,catgeorical_col)






feature_engineering(path).col_remover(catgeorical_col,t1_col_to_encode+t2_col_to_encode+t3_col_to_encode)


feature_engineering(path).col_appender(numerical_col,t1_col_to_encode+t2_col_to_encode+t3_col_to_encode)

Encoder.one_hot_encoder(train_input,catgeorical_col)
encoded_cols=Encoder.encoding_stage2(train_input,catgeorical_col)
Encoder.encoding_stage2(val_input,catgeorical_col)


cols_to_remove=['Functional','Fence','PoolQC','GarageFinish','BsmtFinType2','BsmtFinType1','BsmtExposure','YrSold']

train_input.drop(columns=cols_to_remove,inplace=True)
val_input.drop(columns=cols_to_remove,inplace=True)
for elements in cols_to_remove:
    if elements in numerical_col:
        numerical_col.remove(elements)
    if elements in catgeorical_col:
        catgeorical_col.remove(elements)

train_input[numerical_col]=train_input[numerical_col].fillna(0)
val_input[numerical_col]=val_input[numerical_col].fillna(0)


# Data preprocessing has been completed 


# starting training the model 

class Model(Reading_csv):
    def __init__(self,filepath):
        super().__init__(filepath)
        self.model=LinearRegression()
        
        self.read()

    def Linear_Regression(self,input_for_model,target):
        
        self.model.fit(input_for_model,target)
        return self.model
        

    def model_predictions(self,input_for_model):


        predictions=self.model.predict(input_for_model)
        self.predictions=predictions
        
        featured_weights=pd.DataFrame({


            'Feature':numerical_col+encoded_cols,
            'weights': self.model.coef_
        })
        self.featured_weights=featured_weights
        return predictions, featured_weights
                 


input_for_model=train_input[numerical_col+encoded_cols]



Model_instance=Model(path)
Model_instance.Linear_Regression(input_for_model,train_target)
predictions_tm,featured_weights_tm=Model_instance.model_predictions(input_for_model)

#Model work has been completed on training dataset

# Model prediction for validation dataset


predictions_vm,featured_weights_vm=Model_instance.model_predictions(val_input[numerical_col+encoded_cols])


# prediction with preprocessing for val_input



initial_feature=featured_weights_tm.iloc[0:21]


# create a bargraph about weights for different features
X=initial_feature['Feature']
Y=initial_feature['weights']

pl.figure(figsize=(20,10))
ax=sns.barplot(x=X,y=Y, data=initial_feature)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
pl.show()


class evaluation(Model):
    def __init__(self,filepath):
        super().__init__(filepath)
    def prediction_VS_target(self,X1,Y1,X2,Y2):
        pl.subplot(1,2,1)   # one ffor training
        sns.scatterplot(x=X1,y=Y1)
        pl.title('prediction V/S original target for training dataset')
        pl.subplot(1,2,2)   # one for validation
        sns.scatterplot(x=X2,y=Y2)
        pl.title('prediction V/S original target for validation dataset')
        pl.show()

    def evaluation_metrics(self,given_target,prediction_target):
        mae=mean_absolute_error(given_target,prediction_target)
        mse=mean_squared_error(given_target,prediction_target)
        r2=r2_score(given_target,prediction_target)
        self.mae=mae
        self.mse=mse
        self.r2=r2
        return mae,mse,r2


    def heatmap_for_metrics(self,mae_tm,mse_tm,r2_tm,mae_vm,mse_vm,r2_vm):
        metrics_for_map1=np.array([mae_tm,mse_tm,r2_tm])
        metrics_for_map1=metrics_for_map1.reshape(1,-1)
        metrics_for_map2=np.array([mae_vm,mse_vm,r2_vm])
        metrics_for_map2=metrics_for_map2.reshape(1,-1)

        pl.subplot(1,2,1)
        sns.heatmap(metrics_for_map2,annot=True,xticklabels=['MAE', 'MSE', 'R2'])
        pl.xlabel('Metrics')
        pl.title('Performance Metrics for validation Dataset')

        pl.subplot(1,2,2)
        sns.heatmap(metrics_for_map1,annot=True,xticklabels=['MAE', 'MSE', 'R2'])
        pl.xlabel('Metrics')
        pl.title('Performance Metrics for training Dataset')
        pl.show()





Evaluating=evaluation(path)

Evaluating.prediction_VS_target(predictions_tm,train_target,predictions_vm,val_target)  # scatter plot
mae_tm,mse_tm,r2_tm=Evaluating.evaluation_metrics(train_target,predictions_tm)  # metrics for training dataset
mae_vm,mse_vm,r2_vm=Evaluating.evaluation_metrics(val_target,predictions_vm)    # metrics for validation dataset 


Evaluating.heatmap_for_metrics(mae_tm,mse_tm,r2_tm,mae_vm,mse_vm,r2_vm)


