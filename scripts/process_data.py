import pandas as pd


# Чтение данных из CSV-файла в объект DataFrame
df = pd.read_csv('/home/ivan/Git/MLOps_HW_3S/xflow/datasets/data.csv', header=None)

# Нормализация данных 
df[0] = (df[0] - df[0].min()) / (df[0].max() - df[0].min())

# Запись обработанных данных и перебор значений
with open('/home/ivan/Git/MLOps_HW_3S/xflow/datasets/data_processed.csv', 'w') as f:
    for i, item in enumerate(df[0].values):
        f.write("{},{}\n".format(i, item))
