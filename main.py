import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import os

horizon = 10
month_length = 30
inf = 1000000000
test_size = 28

def load_merge_data(dir) :
    df_train = pd.read_excel(
        dir+"/train.xlsx").rename(columns={"dt": "timestamp", "Цена на арматуру": "target"})
    df_test = pd.read_excel(
        dir+"/test.xlsx").rename(columns={"dt": "timestamp", "Цена на арматуру": "target"})
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    df_chmf = pd.read_csv(
        dir+"/CHMF Акции.csv").rename(columns={"Date": "timestamp"})
    df_magn = pd.read_csv(
        dir+"/MAGN Акции.csv").rename(columns={"Дата": "timestamp"})
    df_nlmk = pd.read_csv(
        dir+"/NLMK Акции.csv").rename(columns={"Date": "timestamp"})
    df_transfer = pd.read_excel(dir+"/Грузоперевозки.xlsx").rename(
        columns={"dt": "timestamp", "Индекс стоимости грузоперевозок": "transfer_cost"})
    df_market = pd.read_excel(
        dir+"/Данные рынка стройматериалов.xlsx").rename(columns={"dt": "timestamp"})
    df_lme = pd.read_excel(
        dir+"/Индекс LME.xlsx").rename(columns={"дата": "timestamp"})
    df_macro = pd.read_excel(
        dir+"/Макропоказатели.xlsx").rename(columns={"dt": "timestamp"})
    df_fuel = pd.read_excel(
        dir+"/Топливо.xlsx").rename(columns={"dt": "timestamp"})
    df_raw_prices = pd.read_excel(
        dir+"/Цены на сырье.xlsx").rename(columns={"dt": "timestamp"})

    df_chmf["timestamp"] = pd.to_datetime(df_chmf["timestamp"])
    df_magn["timestamp"] = pd.to_datetime(df_magn["timestamp"])
    df_nlmk["timestamp"] = pd.to_datetime(df_nlmk["timestamp"])

    # Merge the dataframes
    merged_df = pd.merge(df_full, df_chmf, on='timestamp', how='outer')
    merged_df = pd.merge(merged_df, df_magn, on='timestamp', how='outer')
    merged_df = pd.merge(merged_df, df_nlmk, on='timestamp', how='outer')
    merged_df = pd.merge(merged_df, df_transfer, on='timestamp', how='outer')
    merged_df = pd.merge(merged_df, df_market, on='timestamp', how='outer')
    merged_df = pd.merge(merged_df, df_lme, on='timestamp', how='outer')
    merged_df = pd.merge(merged_df, df_macro, on='timestamp', how='outer')
    merged_df = pd.merge(merged_df, df_fuel, on='timestamp', how='outer')
    merged_df = pd.merge(merged_df, df_raw_prices, on='timestamp', how='outer')

    merged_df = merged_df.sort_values("timestamp")

    return merged_df.sort_values("timestamp")

def prepare_data(merged_df):
    for window in range(3, 2*month_length): #target over a rolling window
        merged_df['EMA'+str(window)] = merged_df['target'].ewm(alpha=2 / (window + 1), adjust=False).mean()
    
    merged_df = merged_df.dropna(subset=["target"])

        

    # Iterate over the columns in the DataFrame
    for column in merged_df.columns:
        if column != "timestamp":
            # Check if the column contains non-numeric values
            if merged_df[column].dtype != float:
                # Extract numeric values using regular expressions
                merged_df[column] = merged_df[column].apply(lambda x: re.findall(
                    r"[-+]?\d*\.\d+|\d+", str(x))[0] if re.findall(r"[-+]?\d*\.\d+|\d+", str(x)) else None)

                # Convert the column to float data type
                merged_df[column] = merged_df[column].astype(float)

    # Convert the "timestamp" column to datetime, if needed
    merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"])

    with open("columns_with_high_nan.txt", "r", encoding='utf8') as file:
        columns_to_drop = file.read().splitlines()

    merged_df = merged_df.drop(columns=columns_to_drop)

    return merged_df

def make_data_blocks(df, shift, window): # make a new dataset with data between weeks <t-shift-window> and <t-shift>, to predict the change of price between week <t> and week <t-shift>
    upgrade_df = df[["timestamp"]].copy()
    rga, rgb = shift, shift+window

    x = np.array(df["target"]-df["target"].shift(rga)) / 100 # TODO
    x = x / (np.abs(x)+1) #SoftSign
    upgrade_df["target"] = x

    for col in df.columns:
        if col !="timestamp":            
            for i in range(rga, rgb):
                col1 = col+str(i)
                if i!=rga:
                    upgrade_df[col1] = df[col].shift(i)-df[col].shift(rga)
                else:
                    upgrade_df[col1] = df[col].shift(i)

    upgrade_df = upgrade_df[horizon:]
    return upgrade_df

def train_test_split(df): #cut test data from the end of the dataset
    return (df[:-test_size].copy().reset_index(drop=True), df[-test_size:].copy().reset_index(drop=True))

class model: #contains the model, and data, prepared for it (different models require different states of data)
    def __init__(self, horizon, window, name, feat_dir, is_full_data_train = False):
        df = prepare_data(load_merge_data(feat_dir))
        upgrade_df = make_data_blocks(df, horizon, window)
        if is_full_data_train:    
            upgrade_train = upgrade_df
            upgrade_test = upgrade_df
        else:        
            upgrade_train, upgrade_test = train_test_split(upgrade_df)
        self.test_y = upgrade_test['target']
        self.test_x = upgrade_test.drop(columns=['target', "timestamp"])
        self.train_y = upgrade_train['target']
        self.train_x = upgrade_train.drop(columns=['target', "timestamp"])
        self.name = name
        self.model = CatBoostRegressor(verbose=0)

    def save(self):
        self.model.save_model(self.name)

    def load(self):        
        self.model.load_model(self.name)

    def fit(self):
        self.model = CatBoostRegressor(verbose=0)#, eval_metric = "R2")
        self.model.fit(self.train_x , self.train_y)

    def predict(self):
        return self.model.predict(self.test_x)

    def true(self):
        return self.test_y
    
def decision_prices(test):
    test = test.set_index('dt')
    tender_price = test['Цена на арматуру']
    decision = test['Объем']
    start_date = test.index.min()
    end_date = test.index.max()

    _results = []
    _active_weeks = 0
    for report_date in pd.date_range(start_date, end_date, freq='W-MON'):
        if _active_weeks == 0:  # Пришла пора нового тендера
            _fixed_price = tender_price.loc[report_date]
            _active_weeks = int(decision.loc[report_date])
        _results.append(_fixed_price)
        _active_weeks += -1
    cost = sum(_results)
    return cost # Возвращаем затраты на периоде

class MetaModel:
    def __init__(self, window, model_path, feat_dir, name):
        self.cb_models = []
        self.risk_value = 0
        self.name = name
        self.cnt_models = horizon-1
        if not os.path.exists(model_path+"/models/"+ name):
            os.mkdir(model_path+"/models/"+ name)
        for i in range(1, horizon):
            self.cb_models.append(model(i, window, model_path+"/models/"+ name + "/cb_model_" + str(i)+".cbm",feat_dir))


    def save(self):
        for _model in self.cb_models:
            _model.save()
        # write risk_value in file
        with open(model_path+"/models/"+self.name+"/"+ self.name+".mmm", "w", encoding='utf8') as file:
            file.write(str(self.risk_value)+"\n")
            file.write(str(self.cnt_models)+"\n")


    def load(self):
        for _model in self.cb_models:
            _model.load()
        with open(model_path+"/models/"+ self.name+"/"+self.name+".mmm", "r", encoding='utf8') as file:
            f = file.read().splitlines()
            self.risk_value, self.cnt_models  = float(f[0]), int(f[1])


    def __find_suitable_index(self, lst):
        for i in range(1, lst.__len__()):
            lst[i]= (lst[0]* self.risk_value+lst[i]*(1- self.risk_value))
        for i, num in enumerate(lst):
            if num < 0:
                return i
        return len(lst)  


    def predict(self):        
        result = []
        step = 0
        models_predictions = [_model.predict() for _model in self.cb_models]
        test_size = models_predictions[0].__len__()
        for day_ind in range(test_size):
            if step == 0:
                segment = [models_predictions[i][day_ind] for i in range(self.cnt_models)]
                step = self.__find_suitable_index(segment) + 1
                result.append(min(step, test_size - day_ind))
            else:
                result.append(0)
            step-=1
        return result


def make_prediction(model_path, features_dir, result_file, purchase_date):
    window = 10 
    final_model = MetaModel(window,model_path, features_dir, "test1")
    final_model.load()
    purchase_date = pd.to_datetime(purchase_date)
    res = pd.read_excel(result_file)
    res['Объём'] = final_model.predict()
    return (res[res['dt']==purchase_date]['Объём']).values[0]



'''model_path = '.'
features_dir = './data'
result_file = features_dir+'/test.xlsx'
purchase_date = '2023-01-30'

print(make_prediction(model_path, features_dir, result_file, purchase_date))'''