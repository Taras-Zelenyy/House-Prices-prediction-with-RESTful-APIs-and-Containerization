import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        # --------------------------------------------------------------------------------------------------------------#
        """
            drop columns
        """
        self.dataset.drop(columns=['PoolQC', 'Fence', 'MiscFeature', 'Alley', 'Id', 'FireplaceQu'], axis=1,
                          inplace=True)

        # --------------------------------------------------------------------------------------------------------------#
        '''
            handle outliers
        '''
        # numerical_cols = [
        #     "MSSubClass", "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
        #     "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
        #     "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
        #     "PoolArea", "MiscVal"
        # ]

        # Z-score threshold for upper and lower limits
        z_score_threshold = 3.0

        def Calculator(column):
            mean = self.dataset[column].mean()
            std = self.dataset[column].std()

            upper_limit = mean + z_score_threshold * std
            lower_limit = mean - z_score_threshold * std

            return upper_limit, lower_limit

        upper_limit, lower_limit = Calculator('MSSubClass')
        self.dataset['MSSubClass'] = np.where(self.dataset['MSSubClass'] > upper_limit, upper_limit, np.where(self.dataset['MSSubClass'] < lower_limit, lower_limit, self.dataset['MSSubClass']))
        upper_limit, lower_limit = Calculator('LotFrontage')
        self.dataset['LotFrontage'] = np.where(self.dataset['LotFrontage'] > upper_limit, upper_limit, np.where(self.dataset['LotFrontage'] < lower_limit, lower_limit, self.dataset['LotFrontage']))
        upper_limit, lower_limit = Calculator('LotArea')
        self.dataset['LotArea'] = np.where(self.dataset['LotArea'] > upper_limit, upper_limit, np.where(self.dataset['LotArea'] < lower_limit, lower_limit, self.dataset['LotArea']))
        upper_limit, lower_limit = Calculator('MasVnrArea')
        self.dataset['MasVnrArea'] = np.where(self.dataset['MasVnrArea'] > upper_limit, upper_limit, np.where(self.dataset['MasVnrArea'] < lower_limit, lower_limit, self.dataset['MasVnrArea']))
        upper_limit, lower_limit = Calculator('BsmtFinSF1')
        self.dataset['BsmtFinSF1'] = np.where(self.dataset['BsmtFinSF1'] > upper_limit, upper_limit, np.where(self.dataset['BsmtFinSF1'] < lower_limit, lower_limit, self.dataset['BsmtFinSF1']))
        upper_limit, lower_limit = Calculator('BsmtFinSF2')
        self.dataset['BsmtFinSF2'] = np.where(self.dataset['BsmtFinSF2'] > upper_limit, upper_limit, np.where(self.dataset['BsmtFinSF2'] < lower_limit, lower_limit, self.dataset['BsmtFinSF2']))
        upper_limit, lower_limit = Calculator('BsmtUnfSF')
        self.dataset['BsmtUnfSF'] = np.where(self.dataset['BsmtUnfSF'] > upper_limit, upper_limit, np.where(self.dataset['BsmtUnfSF'] < lower_limit, lower_limit, self.dataset['BsmtUnfSF']))
        upper_limit, lower_limit = Calculator('TotalBsmtSF')
        self.dataset['TotalBsmtSF'] = np.where(self.dataset['TotalBsmtSF'] > upper_limit, upper_limit, np.where(self.dataset['TotalBsmtSF'] < lower_limit, lower_limit, self.dataset['TotalBsmtSF']))
        upper_limit, lower_limit = Calculator('1stFlrSF')
        self.dataset['1stFlrSF'] = np.where(self.dataset['1stFlrSF'] > upper_limit, upper_limit, np.where(self.dataset['1stFlrSF'] < lower_limit, lower_limit, self.dataset['1stFlrSF']))
        upper_limit, lower_limit = Calculator('2ndFlrSF')
        self.dataset['2ndFlrSF'] = np.where(self.dataset['2ndFlrSF'] > upper_limit, upper_limit, np.where(self.dataset['2ndFlrSF'] < lower_limit, lower_limit, self.dataset['2ndFlrSF']))
        upper_limit, lower_limit = Calculator('LowQualFinSF')
        self.dataset['LowQualFinSF'] = np.where(self.dataset['LowQualFinSF'] > upper_limit, upper_limit, np.where(self.dataset['LowQualFinSF'] < lower_limit, lower_limit, self.dataset['LowQualFinSF']))
        upper_limit, lower_limit = Calculator('GrLivArea')
        self.dataset['GrLivArea'] = np.where(self.dataset['GrLivArea'] > upper_limit, upper_limit, np.where(self.dataset['GrLivArea'] < lower_limit, lower_limit, self.dataset['GrLivArea']))
        upper_limit, lower_limit = Calculator('GarageArea')
        self.dataset['GarageArea'] = np.where(self.dataset['GarageArea'] > upper_limit, upper_limit, np.where(self.dataset['GarageArea'] < lower_limit, lower_limit, self.dataset['GarageArea']))
        upper_limit, lower_limit = Calculator('WoodDeckSF')
        self.dataset['WoodDeckSF'] = np.where(self.dataset['WoodDeckSF'] > upper_limit, upper_limit, np.where(self.dataset['WoodDeckSF'] < lower_limit, lower_limit, self.dataset['WoodDeckSF']))
        upper_limit, lower_limit = Calculator('OpenPorchSF')
        self.dataset['OpenPorchSF'] = np.where(self.dataset['OpenPorchSF'] > upper_limit, upper_limit, np.where(self.dataset['OpenPorchSF'] < lower_limit, lower_limit, self.dataset['OpenPorchSF']))
        upper_limit, lower_limit = Calculator('EnclosedPorch')
        self.dataset['EnclosedPorch'] = np.where(self.dataset['EnclosedPorch'] > upper_limit, upper_limit, np.where(self.dataset['EnclosedPorch'] < lower_limit, lower_limit, self.dataset['EnclosedPorch']))
        upper_limit, lower_limit = Calculator('3SsnPorch')
        self.dataset['3SsnPorch'] = np.where(self.dataset['3SsnPorch'] > upper_limit, upper_limit, np.where(self.dataset['3SsnPorch'] < lower_limit, lower_limit, self.dataset['3SsnPorch']))
        upper_limit, lower_limit = Calculator('ScreenPorch')
        self.dataset['ScreenPorch'] = np.where(self.dataset['ScreenPorch'] > upper_limit, upper_limit, np.where(self.dataset['ScreenPorch'] < lower_limit, lower_limit, self.dataset['ScreenPorch']))
        upper_limit, lower_limit = Calculator('PoolArea')
        self.dataset['PoolArea'] = np.where(self.dataset['PoolArea'] > upper_limit, upper_limit, np.where(self.dataset['PoolArea'] < lower_limit, lower_limit, self.dataset['PoolArea']))
        upper_limit, lower_limit = Calculator('MiscVal')
        self.dataset['MiscVal'] = np.where(self.dataset['MiscVal'] > upper_limit, upper_limit, np.where(self.dataset['MiscVal'] < lower_limit, lower_limit, self.dataset['MiscVal']))
        #
        # Z-score threshold for upper and lower limits
        # z_score_threshold = 3.0
        # # Iterating through the numerical columns to detect and replace outliers
        # for col in numerical_cols:
        #     # Calculating mean and standard deviation
        #     mean = self.dataset[col].mean()
        #     std = self.dataset[col].std()
        #
        #     # Calculating upper and lower limits
        #     upper_limit = mean + z_score_threshold * std
        #     lower_limit = mean - z_score_threshold * std
        #
        #     # Replacing outliers with upper or lower limits
        #     self.dataset.loc[self.dataset[col] > upper_limit, col] = upper_limit
        #     self.dataset.loc[self.dataset[col] < lower_limit, col] = lower_limit

        # --------------------------------------------------------------------------------------------------------------#
        '''
            fill Nan with median
        '''
        # columns_to_fill_median = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']

        # Apply the function to each column
        self.dataset['LotFrontage'].fillna(self.dataset['LotFrontage'].median(), inplace=True)
        self.dataset['MasVnrArea'].fillna(self.dataset['MasVnrArea'].median(), inplace=True)
        self.dataset['GarageYrBlt'].fillna(self.dataset['GarageYrBlt'].median(), inplace=True)
        # for column in columns_to_fill_median:
        #     self.dataset[column].fillna(self.dataset[column].median(), inplace=True)

        '''
            fill Nan with mode
        '''
        # columns_to_fill_mode = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        #                         'BsmtFinType2', 'Electrical', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

        self.dataset['MasVnrType'] = self.dataset['MasVnrType'].fillna(self.dataset['MasVnrType'].mode()[0])
        self.dataset['BsmtQual'] = self.dataset['BsmtQual'].fillna(self.dataset['BsmtQual'].mode()[0])
        self.dataset['BsmtCond'] = self.dataset['BsmtCond'].fillna(self.dataset['BsmtCond'].mode()[0])
        self.dataset['BsmtExposure'] = self.dataset['BsmtExposure'].fillna(self.dataset['BsmtExposure'].mode()[0])
        self.dataset['BsmtFinType1'] = self.dataset['BsmtFinType1'].fillna(self.dataset['BsmtFinType1'].mode()[0])
        self.dataset['BsmtFinType2'] = self.dataset['BsmtFinType2'].fillna(self.dataset['BsmtFinType2'].mode()[0])
        self.dataset['Electrical'] = self.dataset['Electrical'].fillna(self.dataset['Electrical'].mode()[0])
        self.dataset['GarageType'] = self.dataset['GarageType'].fillna(self.dataset['GarageType'].mode()[0])
        self.dataset['GarageFinish'] = self.dataset['GarageFinish'].fillna(self.dataset['GarageFinish'].mode()[0])
        self.dataset['GarageQual'] = self.dataset['GarageQual'].fillna(self.dataset['GarageQual'].mode()[0])
        self.dataset['GarageCond'] = self.dataset['GarageCond'].fillna(self.dataset['GarageCond'].mode()[0])

        # for col in columns_to_fill_mode:
        #     self.dataset[col] = self.dataset[col].fillna(self.dataset[col].mode()[0])

        '''
            encode labels
        '''
        le = LabelEncoder()

        # categorical_cols = [
        #     "MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope",
        #     "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
        #     "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual",
        #     "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir",
        #     "Electrical", "KitchenQual", "Functional", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        #     "PavedDrive", "SaleType", "SaleCondition"
        # ]

        self.dataset["MSZoning"] = le.fit_transform(self.dataset["MSZoning"])
        self.dataset["Street"] = le.fit_transform(self.dataset["Street"])
        self.dataset["LotShape"] = le.fit_transform(self.dataset["LotShape"])
        self.dataset["LandContour"] = le.fit_transform(self.dataset["LandContour"])
        self.dataset["Utilities"] = le.fit_transform(self.dataset["Utilities"])
        self.dataset["LotConfig"] = le.fit_transform(self.dataset["LotConfig"])
        self.dataset["LandSlope"] = le.fit_transform(self.dataset["LandSlope"])

        self.dataset["Neighborhood"] = le.fit_transform(self.dataset["Neighborhood"])
        self.dataset["Condition1"] = le.fit_transform(self.dataset["Condition1"])
        self.dataset["Condition2"] = le.fit_transform(self.dataset["Condition2"])
        self.dataset["BldgType"] = le.fit_transform(self.dataset["BldgType"])
        self.dataset["HouseStyle"] = le.fit_transform(self.dataset["HouseStyle"])
        self.dataset["RoofStyle"] = le.fit_transform(self.dataset["RoofStyle"])
        self.dataset["RoofMatl"] = le.fit_transform(self.dataset["RoofMatl"])

        self.dataset["Exterior1st"] = le.fit_transform(self.dataset["Exterior1st"])
        self.dataset["Exterior2nd"] = le.fit_transform(self.dataset["Exterior2nd"])
        self.dataset["MasVnrType"] = le.fit_transform(self.dataset["MasVnrType"])
        self.dataset["ExterQual"] = le.fit_transform(self.dataset["ExterQual"])
        self.dataset["ExterCond"] = le.fit_transform(self.dataset["ExterCond"])
        self.dataset["Foundation"] = le.fit_transform(self.dataset["Foundation"])
        self.dataset["BsmtQual"] = le.fit_transform(self.dataset["BsmtQual"])

        self.dataset["BsmtCond"] = le.fit_transform(self.dataset["BsmtCond"])
        self.dataset["BsmtExposure"] = le.fit_transform(self.dataset["BsmtExposure"])
        self.dataset["BsmtFinType1"] = le.fit_transform(self.dataset["BsmtFinType1"])
        self.dataset["BsmtFinType2"] = le.fit_transform(self.dataset["BsmtFinType2"])
        self.dataset["Heating"] = le.fit_transform(self.dataset["Heating"])
        self.dataset["HeatingQC"] = le.fit_transform(self.dataset["HeatingQC"])
        self.dataset["CentralAir"] = le.fit_transform(self.dataset["CentralAir"])

        self.dataset["Electrical"] = le.fit_transform(self.dataset["Electrical"])
        self.dataset["KitchenQual"] = le.fit_transform(self.dataset["KitchenQual"])
        self.dataset["Functional"] = le.fit_transform(self.dataset["Functional"])
        self.dataset["GarageType"] = le.fit_transform(self.dataset["GarageType"])
        self.dataset["GarageFinish"] = le.fit_transform(self.dataset["GarageFinish"])
        self.dataset["GarageQual"] = le.fit_transform(self.dataset["GarageQual"])
        self.dataset["GarageCond"] = le.fit_transform(self.dataset["GarageCond"])

        self.dataset["PavedDrive"] = le.fit_transform(self.dataset["PavedDrive"])
        self.dataset["SaleType"] = le.fit_transform(self.dataset["SaleType"])
        self.dataset["SaleCondition"] = le.fit_transform(self.dataset["SaleCondition"])

        # for col in categorical_cols:
        #     self.dataset[col] = le.fit_transform(self.dataset[col])

        '''
            normalization
        '''
        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()

        # Apply the scaler to the processed numerical data
        # normalization_cols = [
        #     "MSSubClass", "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
        #     "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
        #     "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
        #     "PoolArea", "MiscVal"
        # ]
        temp_dataset = self.dataset.copy()
        self.dataset["MSSubClass"] = scaler.fit_transform(temp_dataset[["MSSubClass"]])
        self.dataset["LotFrontage"] = scaler.fit_transform(temp_dataset[["LotFrontage"]])
        self.dataset["LotArea"] = scaler.fit_transform(temp_dataset[["LotArea"]])
        self.dataset["MasVnrArea"] = scaler.fit_transform(temp_dataset[["MasVnrArea"]])
        self.dataset["BsmtFinSF1"] = scaler.fit_transform(temp_dataset[["BsmtFinSF1"]])
        self.dataset["BsmtFinSF2"] = scaler.fit_transform(temp_dataset[["BsmtFinSF2"]])
        self.dataset["BsmtUnfSF"] = scaler.fit_transform(temp_dataset[["BsmtUnfSF"]])
        self.dataset["TotalBsmtSF"] = scaler.fit_transform(temp_dataset[["TotalBsmtSF"]])
        self.dataset["1stFlrSF"] = scaler.fit_transform(temp_dataset[["1stFlrSF"]])
        self.dataset["2ndFlrSF"] = scaler.fit_transform(temp_dataset[["2ndFlrSF"]])
        self.dataset["LowQualFinSF"] = scaler.fit_transform(temp_dataset[["LowQualFinSF"]])
        self.dataset["GrLivArea"] = scaler.fit_transform(temp_dataset[["GrLivArea"]])
        self.dataset["GarageArea"] = scaler.fit_transform(temp_dataset[["GarageArea"]])
        self.dataset["WoodDeckSF"] = scaler.fit_transform(temp_dataset[["WoodDeckSF"]])
        self.dataset["OpenPorchSF"] = scaler.fit_transform(temp_dataset[["OpenPorchSF"]])
        self.dataset["EnclosedPorch"] = scaler.fit_transform(temp_dataset[["EnclosedPorch"]])
        self.dataset["3SsnPorch"] = scaler.fit_transform(temp_dataset[["3SsnPorch"]])
        self.dataset["ScreenPorch"] = scaler.fit_transform(temp_dataset[["ScreenPorch"]])
        self.dataset["PoolArea"] = scaler.fit_transform(temp_dataset[["PoolArea"]])
        self.dataset["MiscVal"] = scaler.fit_transform(temp_dataset[["MiscVal"]])

        # for col in normalization_cols:
        #     self.dataset[col] = scaler.fit_transform(temp_dataset[[col]])

        return self.dataset
