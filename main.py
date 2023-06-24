import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
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

    nan_counts = merged_df.isna().sum()
    columns_with_high_nan = nan_counts[nan_counts > merged_df.shape[0] * 0.5].index
    merged_df = merged_df.drop(columns=columns_with_high_nan)

    #TODO:fillna

    return merged_df
def make_data_blocks(df, shift, sz): # make a new dataset with data between weeks <t-shift-sz> and <t-shift>, to predict the change of price between week <t> and week <t-shift>
    upgrade_df = df[["timestamp"]].copy()
    rga, rgb = shift, shift+sz

    x = np.array(df["target"]-df["target"].shift(rga)) / 100
    x = x / (np.abs(x)+1) #SoftSign
    upgrade_df["target"] = x

    for col in df.columns:
        if col not in ["timestamp"]:
            for i in range(rga, rgb):
                col1 = col+str(i)
                if i!=rga:
                    upgrade_df[col1] = df[col].shift(i)-df[col].shift(rga)
                else:
                    upgrade_df[col1] = df[col].shift(i)

    upgrade_df = upgrade_df[10:]
    return upgrade_df

test_size = 28

def train_test_split(df): #cut test data from the end of the dataset
    return (df[:-test_size].copy().reset_index(drop=True), df[-test_size:].copy().reset_index(drop=True))
class model: #contains the model, and data, prepared for it (different models require different states of data)
    def __init__(self, shift, sz, name, featdir):
        df = prepare_data(load_merge_data(featdir))
        upgrade_df = make_data_blocks(df, shift, sz)
        upgrade_train, upgrade_test = train_test_split(upgrade_df)
        self.test_y = upgrade_test['target']
        self.test_x = upgrade_test.drop(columns=['target', "timestamp"])
        self.train_y = upgrade_train['target']
        self.train_x = upgrade_train.drop(columns=['target', "timestamp"])
        self.name = name

    def save(self):
        self.model.save_model(self.name)

    def load(self):
        self.model = CatBoostRegressor(verbose=0)
        self.model.load_model(self.name)

    def fit(self):
        self.model = CatBoostRegressor(verbose=0)#, eval_metric = "R2")
        self.model.fit(self.train_x , self.train_y)

    def predict(self):
        return self.model.predict(self.test_x)

    def true(self):
        return self.test_y
def find_negative_index(lst):
    for i, num in enumerate(lst):
        if num < 0:
            return i
    return len(lst)

def make_orders_on_time_segment(models):
    result = []
    step = 0
    models_predictions = [model.predict() for model in models]
    for day_ind in range(test_size):
        if step == 0:
            segment = [models_predictions[i][day_ind] for i in range(len(models))]
            step = find_negative_index(segment)+1
            result.append(min(step, test_size-day_ind))
        else:
            result.append(0)
        step-=1
    return result
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
block_sz = 3 #hyperparam 3 is outputted in the train notebook (we didn't save it in the model file yet) (and it is probably wrong)


def make_prediction(model_path, features_dir, result_file, purchase_date):
    models = []
    for i in range(1, 10):
        models.append(model(i, block_sz, model_path+"/models/cb_model_" + str(i), features_dir))
        models[-1].load()
    purchase_date = pd.to_datetime(purchase_date)
    res = pd.read_excel(result_file)
    res['Объём'] = make_orders_on_time_segment(models)
    return res[res['dt']==purchase_date]['Объём']

'''model_path = 'path'
features_dir = 'path'
result_file = 'test.xlsx'
purchase_date = '2023-05-17'

make_prediction(model_path, features_dir, result_file, purchase_date)'''