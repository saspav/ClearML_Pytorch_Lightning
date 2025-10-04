import pandas as pd

from custom_classes import CFG

cfg = CFG()

# Загружаем списки train/val и test
try:
    with open(cfg.data_dir / "train_val_list.txt") as f:
        train_val_list = set(x.strip() for x in f.readlines())
except FileNotFoundError:
    print(f'❌ Загрузите файл "train_val_list.txt"')

try:
    with open(cfg.data_dir / "test_list.txt") as f:
        test_list = set(x.strip() for x in f.readlines())
except FileNotFoundError:
    print(f'❌ Загрузите файл "test_list.txt"')

# Загружаем bbox + labels
try:
    meta_df = pd.read_csv(cfg.data_dir / "Data_Entry_2017_v2020.csv").rename(
        columns={'Finding Labels': "Finding Label"})
except FileNotFoundError:
    print(f'❌ Загрузите файл "Data_Entry_2017_v2020.txt"')

# Берём только картинки, реально существующие в images/
existing_images = {p.name for p in cfg.images_dir.glob("*.png")}
bbox_df = meta_df[meta_df["Image Index"].isin(existing_images)]

# Готовим датасет
df = pd.DataFrame(data=sorted(existing_images), columns=["Image Index"])
df = df.merge(bbox_df.drop_duplicates(subset=["Image Index"]), on="Image Index", how="left")
# Заполняем пропуски
df["Finding Label"] = df["Finding Label"].fillna("No Finding")

# Формируем таргет
df["target"] = (df["Finding Label"] != "No Finding").astype(int)
print('Распределение целевой переменной')
print(df.target.value_counts())

# Признаки для модели
export_cols = ['Image Index', 'Finding Label', 'target']
df = df[export_cols]

# Разделение по спискам
train_val_df = df[df["Image Index"].isin(train_val_list)]
test_df = df[df["Image Index"].isin(test_list)]

# Проверка
print("Train/Val shape:", train_val_df.shape)
print("Test shape:", test_df.shape)

# Сохраняем
train_val_df.to_csv(cfg.data_dir / cfg.train_csv, index=False)
test_df.to_csv(cfg.data_dir / cfg.test_csv, index=False)
