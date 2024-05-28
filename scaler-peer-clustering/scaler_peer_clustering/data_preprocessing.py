import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

class DataPreprocessor():
    def __init__(self, df):
        self.df = df
        self.cat_cols = [col for col in self.df.columns if self.df[col].dtype == "O" or self.df[col].nunique() <= 30]

    def org_year(self):
        # median_value = self.df.orgyear.median()
        n = self.df["orgyear"].isna().sum()
        self.df["orgyear"].fillna(self.df.groupby("company_hash")["orgyear"].transform("median"), inplace=True)
        self.df = self.df.loc[~(self.df['orgyear'].isna())]
        # self.df.loc[self.df.orgyear.isna(), "orgyear"] = median_value
        self.df["orgyear"] = self.df["orgyear"].clip(lower=self.df.orgyear.quantile(0.01), upper=self.df.orgyear.quantile(0.99))
        print("missing values {} are filled in orgyear, and outlirs are clipped".format(n))
        return self
    
    def ctc(self):
        n = self.df["ctc"].isna().sum()
        self.df["ctc"] = self.df["ctc"].clip(lower=self.df.ctc.quantile(0.01), upper=self.df.ctc.quantile(0.99))
        print("missing values {} are filled in ctc, and outlirs are clipped".format(n))
        return self
    
    def job_position(self):
        n = self.df["job_position"].isna().sum()
        self.df["job_position"] = self.df.job_position.fillna("Unknown")
        print("missing values {} are filled in job_position to 'Unknown', and shape = {}".format(n, self.df.shape))
        return self
        
    def hike_in_current_org(self):
        self.df["hike_in_current_org"] = (self.df["orgyear"] < self.df["ctc_updated_year"]).astype("int")
        print("'hike_in_current_org', feature is created")
        return self
    
    def company_hash(self):
        n = self.df["company_hash"].isna().sum()
        self.df["company_hash"] = self.df.company_hash.fillna("Unknown")
        print("missing values {} are filled in company_hash to 'Unknown', and shape = {}".format(n, self.df.shape))
        return self

    def categorical_encoding(self, column):
        frequency_map = self.df[column].value_counts(normalize=True)
        self.df[column] = self.df[column].map(frequency_map)
        print("frequency mapping ecoding done for column {}, and shape = {}".format(column, self.df.shape))
        return self
    
    def scaling_data(self):
        scaler = StandardScaler()
        scaler.fit(self.df)
        self.scaled_data = scaler.transform(self.df)
        self.df = pd.DataFrame(self.scaled_data, columns = self.df.columns)
        print("df is scaled (standard scaler)")
        return self
    
    def return_df(self):
        return self.df
    
    def transform(self, c_encoding = True):
        print("shape before pipeline = {}".format(self.df.shape))
        self.org_year()
        self.ctc()
        self.job_position()
        self.hike_in_current_org()
        self.company_hash()
        if c_encoding == True:
            self.categorical_encoding("company_hash")
            self.categorical_encoding("job_position")
            self.categorical_encoding("hike_in_current_org")
            self.scaling_data()
        
        print("any duplicates present? {}, {}".format(self.df.duplicated().sum() > 0, self.df.duplicated().sum()))
        self.df.drop_duplicates(keep = "first", inplace = True)
        print("duplicates values are dropped, and shape = {}".format(self.df.shape))
        print("shape after pipeline = {}".format(self.df.shape))

        return self.return_df()


        
