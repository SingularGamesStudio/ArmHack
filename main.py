import pandas as pd

def make_prediction(model_path, features_dir, result_file, purchase_date):
    purchase_date = pd.to_datetime(purchase_date)
    res = pd.read_excel(result_file)
    return res[res['dt']==purchase_date]['Объём']

'''model_path = 'path'
features_dir = 'path'
result_file = 'test.xlsx'
purchase_date = '2023-05-17'

make_prediction(model_path, features_dir, result_file, purchase_date)'''