# -*- coding: utf-8 -*-
"""
train_lightning_signlang.py

Перенос обучения свёрточной сети для Sign Language MNIST на PyTorch Lightning
с поддержкой fast_dev_run, ClearML и визуализации метрик.
"""

import os
import argparse
import sys
import time
import zipfile
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt

from clearml import Task, Logger
from getpass import getpass
from tqdm import tqdm


# ===================== CFG ClearML =====================
@dataclass
class CFG:
    """
    Класс для хранения всех параметров конфигурации эксперимента.

    Атрибуты:
        project_name (str): Название проекта в ClearML.
        experiment_name (str): Название конкретного эксперимента в ClearML.
        test_size (float): Часть под валидационный датасет
        seed (int): Фиксированный seed для воспроизводимости.
        batch_size (int): Размер батча для обучения/валидации.
        max_epochs (int): Количество эпох для обучения модели.
        num_workers (int): Количество воркеров для загрузки данных.
        data_url_train (str): Ссылка на архив с тренировочным датасетом.
        data_url_test (str): Ссылка на архив с тестовым датасетом.
        data_dir (Path): Папка, куда будут скачаны и распакованы данные.
    """
    project_name: str = "SignLanguage Project"
    experiment_name: str = "Lightning CNN"
    test_size: float = 0.25
    seed: int = 127
    batch_size: int = 512
    max_epochs: int = 30
    num_workers: int = 24
    data_url_train: str = "https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/sign_mnist_train.csv.zip"
    data_url_test: str = "https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/sign_mnist_test.csv.zip"
    data_dir: Path = Path("data")


# ===================== Dataset =====================
class SignLanguageDataset(Dataset):
    """
    Кастомный Dataset для Sign Language MNIST.

    Каждая строка CSV состоит из:
        - первого столбца: метка класса (int, от 0 до 25),
        - остальных столбцов: значения пикселей изображения 28x28.

    Аргументы:
        df (pd.DataFrame): DataFrame с данными (train или test).
        transform (torchvision.transforms): Аугментации или преобразования изображений.

    Методы:
        __len__: возвращает количество элементов в датасете.
        __getitem__: возвращает (изображение, метка) по индексу.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        # Сохраняем dataframe и сбрасываем индекс для надёжности
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        # Размер датасета равен числу строк в DataFrame
        return len(self.df)

    def __getitem__(self, idx):
        # Берём строку по индексу
        row = self.df.iloc[idx]
        # Первый столбец — это метка класса
        label = int(row.iloc[0])
        # Остальные значения — это пиксели 28x28
        pixels = row.iloc[1:].values.astype(np.float32).reshape(28, 28)
        # Добавляем ось канала (1, 28, 28), чтобы соответствовать Conv2D
        img = torch.tensor(pixels).unsqueeze(0)
        # Применяем аугментации, если они заданы
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ===================== DataModule =====================
class SignLanguageDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule для Sign Language MNIST.

    Этот класс инкапсулирует всю логику подготовки и загрузки данных:
    - определяет аугментации для обучения,
    - создает кастомные Dataset,
    - возвращает готовые DataLoader для обучения, валидации и теста.

    Аргументы:
        train_csv (str): путь к CSV-файлу с тренировочными данными.
        test_csv (str): путь к CSV-файлу с тестовыми данными.
        batch_size (int): размер батча.
        num_workers (int): число потоков для DataLoader (по умолчанию 8).

    Методы:
        setup(stage=None): загружает CSV-файлы и создаёт Dataset для train/val/test.
        train_dataloader(): возвращает DataLoader для обучения.
        val_dataloader(): возвращает DataLoader для валидации.
        test_dataloader(): возвращает DataLoader для тестирования.
    """

    def __init__(self, train_csv: str, test_csv: str, batch_size: int, num_workers: int = 8):
        super().__init__()
        self.train_dataset = self.valid_dataset = self.test_dataset = None
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Аугментации для обучения:
        # - случайное отражение по горизонтали (10%)
        # - случайный поворот (20%)
        self.train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.1),
             transforms.RandomApply([transforms.RandomRotation(degrees=(-180, 180))], p=0.2),
             ]
        )

        # Для теста и валидации аугментации не нужны
        self.test_transform = None

    def setup(self, stage=None):
        """
        Загружает CSV-файлы и создает объекты Dataset.
        Lightning автоматически вызывает этот метод в начале fit/test.
        """
        train_df = pd.read_csv(self.train_csv)
        test_df = pd.read_csv(self.test_csv)

        # Поделим трейн на обучение и валидацию
        train, valid = train_test_split(train_df,
                                        test_size=cfg.test_size,
                                        random_state=cfg.seed,
                                        )
        # Создаём датасеты
        self.train_dataset = SignLanguageDataset(train, transform=self.train_transform)
        self.valid_dataset = SignLanguageDataset(valid, transform=self.test_transform)
        self.test_dataset = SignLanguageDataset(test_df, transform=self.test_transform)

    def train_dataloader(self):
        """DataLoader для обучения (с перемешиванием и аугментациями)."""
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          )

    def val_dataloader(self):
        """DataLoader для валидации (без перемешивания)."""
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          )

    def test_dataloader(self):
        """DataLoader для теста (без перемешивания)."""
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          )


# ===================== Model =====================
class MyConvNet(nn.Module):
    """
    Простая сверточная нейросеть для классификации изображений
    (например, датасет жестового языка, 28x28, 1 канал).

    Архитектура:
        - Блок 1: Conv2d -> BatchNorm -> AvgPool -> ReLU
        - Блок 2: Conv2d -> BatchNorm -> AvgPool -> ReLU
        - Полносвязные слои: Linear -> LeakyReLU -> Dropout -> Linear

    Аргументы:
        n_classes (int): количество выходных классов (по умолчанию 25).
    """

    def __init__(self, n_classes=25):
        super().__init__()

        # Первый сверточный блок:
        # - Conv2d: 1 канал -> 8 фильтров (3x3), паддинг=1 для сохранения размера
        # - BatchNorm2d: нормализация по каналам
        # - AvgPool2d: уменьшение размерности в 2 раза
        # - ReLU: активация
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.AvgPool2d(2),
            nn.ReLU(),
        )

        # Второй сверточный блок:
        # - Conv2d: 8 каналов -> 16 фильтров (3x3), паддинг=1
        # - BatchNorm2d
        # - AvgPool2d: ещё раз уменьшаем размерность в 2 раза
        # - ReLU
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(2),
            nn.ReLU(),
        )

        self.lin1 = nn.Linear(16 * 7 * 7, 100)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(0.3)
        self.lin2 = nn.Linear(100, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.lin2(x)
        return x


class LitSignLang(pl.LightningModule):
    """
    LightningModule для обучения сверточной сети на датасете жестового языка.

    Содержит:
      - Модель (MyConvNet)
      - Функцию потерь (CrossEntropyLoss)
      - Логику обучения и валидации
      - Интеграцию с ClearML для логирования метрик

    Аргументы:
        lr (float): скорость обучения (по умолчанию 1e-3)
        n_classes (int): количество классов (по умолчанию 25)
    """

    def __init__(self, lr: float = 1e-3, n_classes: int = 25):
        super().__init__()
        # Cохраняем гиперпараметры в атрибуты
        self.lr = lr
        self.n_classes = n_classes
        self.save_hyperparameters()
        # Основная модель — наша сверточная сеть
        self.model = MyConvNet(n_classes=n_classes)
        # Кросс-энтропия — стандартная функция потерь для классификации
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Прямой проход через модель.
        Вход:  x — тензор изображений
        Выход: логиты классов
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Один шаг обучения:
          - считаем loss
          - предсказываем классы
          - считаем accuracy
          - логируем метрики в Lightning и ClearML
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Логирование в Lightning
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)

        # Логирование в ClearML
        logger_clearml.report_scalar("train_loss", "epoch", loss.item(), self.current_epoch)
        logger_clearml.report_scalar("train_acc", "epoch", acc.item(), self.current_epoch)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Один шаг валидации:
          - считаем loss
          - предсказываем классы
          - считаем accuracy
          - логируем метрики в Lightning и ClearML
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Логирование в Lightning (отображение в прогресс-баре)
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True)
        self.log("valid_acc", acc, on_epoch=True, prog_bar=True)

        # Логирование в ClearML
        logger_clearml.report_scalar("valid_loss", "epoch", loss.item(), self.current_epoch)
        logger_clearml.report_scalar("valid_acc", "epoch", acc.item(), self.current_epoch)

    def configure_optimizers(self):
        """
        Конфигурация оптимизатора.
        Используется Adam с lr из self.lr.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ===================== Load Dataset =====================
def download_and_extract(url: str, save_dir: Path):
    """
    Скачивает и распаковывает zip-архив с датасетом по указанному URL
    и логирует процесс в ClearML.

    Аргументы:
        url (str): ссылка на архив с датасетом
        save_dir (Path): путь к папке, куда сохранять файл и распаковывать содержимое

    Логика:
        - Создаёт директорию, если её нет
        - Скачивает архив, если он ещё не существует
        - Распаковывает содержимое архива в указанную папку
        - Логирует все этапы в ClearML
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / Path(url).name

    if not filename.exists():
        msg = f"Скачиваю датасет: {url}"
        logger_clearml.report_text(msg)
        urllib.request.urlretrieve(url, filename)
    else:
        msg = f"Файл {filename} уже существует, пропускаю скачивание."
        logger_clearml.report_text(msg)

    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(save_dir)

    msg = f"Файл {filename} успешно распакован в {save_dir}"
    logger_clearml.report_text(msg)


def maybe_load_dataset(load_dataset: bool):
    """
    Проверяет наличие датасета и при необходимости скачивает его.

    Аргументы:
        load_dataset (bool): если True, то скачивает и распаковывает датасеты заново

    Возвращает:
        (str, str): пути к train_csv и test_csv

    Исключения:
        FileNotFoundError — если датасеты отсутствуют и не удалось их скачать
    """
    train_csv = cfg.data_dir / "sign_mnist_train.csv"
    test_csv = cfg.data_dir / "sign_mnist_test.csv"

    # Если указано явно — скачиваем и распаковываем
    if load_dataset:
        download_and_extract(cfg.data_url_train, cfg.data_dir)
        download_and_extract(cfg.data_url_test, cfg.data_dir)

    # Проверка, что файлы существуют
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("Не найдены датасеты. "
                                "Запустите скрипт с параметром --load_dataset True")

    return str(train_csv), str(test_csv)


def make_datamodule(args):
    """
    Загружаем пути к train/test CSV (при необходимости скачиваем)
    :param args: аргументы командной строки (включая load_dataset и fast_dev_run)
    :return: экземпляр класса SignLanguageDataModule
    """
    train_csv, test_csv = maybe_load_dataset(args.load_dataset)
    datamodule = SignLanguageDataModule(train_csv, test_csv, batch_size=cfg.batch_size)
    datamodule.setup()
    return datamodule


# ===================== Training =====================
def run_training(args, datamodule: SignLanguageDataModule):
    """
    Основной процесс обучения модели с поддержкой:
      - загрузки/скачивания датасета
      - fast_dev_run (быстрый тестовый прогон)
      - обучения на GPU/CPU
      - логирования метрик в CSVLogger и ClearML
      - сохранения модели
      - построения и отправки графиков обучения в ClearML
      - тестового инференса на одном примере

    Аргументы:
        args: аргументы командной строки (включая load_dataset и fast_dev_run)
        datamodule: экземпляр класса SignLanguageDataModule
    """
    # Загружаем пути к train/test CSV (при необходимости скачиваем)
    train_csv, test_csv = maybe_load_dataset(args.load_dataset)
    dm = SignLanguageDataModule(train_csv, test_csv, batch_size=cfg.batch_size)
    dm.setup()

    # Оптимизация умножений матриц (PyTorch 2.0+)
    try:
        torch.set_float32_matmul_precision('high')
    except:
        pass

    # Инициализация модели
    model = LitSignLang()

    # Логгер Lightning (сохраняет метрики в CSV)
    csv_logger = CSVLogger("lightning_logs", name="signlang_experiment")

    # Определение устройства
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # ===================== Быстрый тестовый прогон =====================
    if args.fast_dev_run:
        print("Тестовый прогон...")
        trainer = pl.Trainer(accelerator=accelerator, fast_dev_run=True, logger=csv_logger)
        try:
            trainer.fit(model, datamodule)
            print("\nТестовый прогон успешно пройден")
        except Exception as e:
            print("\nТестовый прогон завершился с ошибкой:", e)
            print("✅ Завершаю Task'у...")
            task.close()
            sys.exit(1)

    # ===================== Основное обучение =====================
    # чекпоинт для сохранения лучшей модели
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename="best_model",
        save_top_k=1,
        # monitor="valid_loss",  # метрика для раннего стопа
        # mode="min",  # минимизируем valid_loss
        monitor="valid_acc",  # метрика для раннего стопа
        mode="max",  # максимизируем valid_acc
        save_weights_only=True  # сохраняем только веса
    )

    # ранний стоп
    early_stopping_callback = EarlyStopping(
        patience=5,  # если не улучшается 5 эпох подряд — останавливаем
        # monitor="valid_loss",
        # mode="min",
        monitor="valid_acc",
        mode="max",
        verbose=True
    )

    trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                         accelerator=accelerator,
                         logger=csv_logger,
                         deterministic=True,
                         # добавляем колбэки
                         callbacks=[checkpoint_callback, early_stopping_callback]
                         )

    start = time.time()
    trainer.fit(model, datamodule)
    print(f"Обучение завершено за {time.time() - start:.1f} секунд")

    # ===================== Сохранение весов =====================
    ckpt_path = MODEL_DIR / "myconvnet_sign_lang.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    print(f"Модель сохранена: {ckpt_path}")

    # ===================== Визуализация кривых обучения =====================
    metrics = pd.read_csv(csv_logger.log_dir + "/metrics.csv")

    plt.figure(figsize=(12, 5))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_loss_epoch"].dropna(), label="train_loss")
    plt.plot(metrics["valid_loss"].dropna(), label="valid_loss")
    plt.legend()
    plt.title("Loss")

    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(metrics["train_acc_epoch"].dropna(), label="train_acc")
    plt.plot(metrics["valid_acc"].dropna(), label="valid_acc")
    plt.legend()
    plt.title("Accuracy")

    # Сохраняем график
    fig_path = MODEL_DIR / "training_curves.png"
    plt.savefig(fig_path)
    plt.close()

    # ===================== Отправка графика в ClearML =====================
    logger = Logger.current_logger()
    logger.report_image(
        title="Learning Curves",
        series="Logloss",
        local_path=str(fig_path),
    )
    return str(ckpt_path)


# ===================== Main =====================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast_dev_run", type=lambda x: str(x).lower() in ["true", "1"],
                        default=False)
    parser.add_argument("--load_dataset", type=lambda x: str(x).lower() in ["true", "1"],
                        default=False)
    parser.add_argument("--inference_test", type=lambda x: str(x).lower() in ["true", "1"],
                        default=False)
    return parser.parse_args()


def check_clearml_env():
    required_env_vars = [
        "CLEARML_WEB_HOST",
        "CLEARML_API_HOST",
        "CLEARML_FILES_HOST",
        "CLEARML_API_ACCESS_KEY",
        "CLEARML_API_SECRET_KEY"
    ]
    env_vars = dict(CLEARML_WEB_HOST="https://app.clear.ml/",
                    CLEARML_API_HOST="https://api.clear.ml",
                    CLEARML_FILES_HOST="https://files.clear.ml"
                    )

    for var in required_env_vars[:3]:
        if os.getenv(var) is None:
            os.environ[var] = env_vars[var]

    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]

    if missing_vars:
        print("⚠️  Некоторые переменные среды ClearML отсутствуют.")
        for var in missing_vars:
            os.environ[var] = getpass(f"Введите значение для {var}: ")
        print("✅ Все переменные ClearML установлены.\n")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def inference_on_test(ckpt_path: str, datamodule: SignLanguageDataModule):
    """
    Инференс на всем тестовом датасете с вычислением accuracy и логированием в ClearML.

    Args:
        ckpt_path (str): путь к сохранённой модели (.ckpt)
        datamodule (SignLanguageDataModule): подготовленный DataModule с тестовым датасетом
    """
    # Загружаем лучшую модель
    model = LitSignLang.load_from_checkpoint(ckpt_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Получаем DataLoader для теста
    test_loader = datamodule.test_dataloader()

    all_preds = []
    all_labels = []

    # Инференс на всем тестовом датасете
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Inference on test set", unit="batch"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Вычисляем accuracy
    all_preds_tensor = torch.tensor(all_preds)
    all_labels_tensor = torch.tensor(all_labels)
    accuracy = (all_preds_tensor == all_labels_tensor).float().mean().item()
    print(f"Accuracy на тесте: {accuracy:.4f}")

    # Логируем accuracy в ClearML
    logger_clearml.report_scalar("Test Accuracy", "Accuracy", accuracy, 0)

    # Отображаем первые 5 примеров на одной фигуре
    num_examples = min(5, len(all_preds))
    fig, axes = plt.subplots(1, num_examples, figsize=(3 * num_examples, 3))

    for i, ax in enumerate(axes):
        img, true_label, pred_label = test_loader.dataset[i][0], all_labels[i], all_preds[i]
        ax.imshow(img.squeeze().numpy(), cmap="gray")
        ax.set_title(f"T:{true_label}, P:{pred_label}")
        ax.axis("off")

    fig_path = MODEL_DIR / "inference_examples_grid.png"
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)

    # Логируем в ClearML
    logger_clearml.report_image(
        title="Inference Examples",
        series="Examples",
        local_path=str(fig_path)
    )


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    args = parse_args()  # парсим аргументы

    # ===================== Init ClearML =====================
    cfg = CFG()
    save_cfg = cfg.__dict__.copy()
    save_cfg['data_dir'] = str(save_cfg['data_dir'])

    # создаём Task (важно, чтобы он был виден глобально)
    task = Task.init(project_name=cfg.project_name,
                     task_name=cfg.experiment_name,
                     task_type=Task.TaskTypes.training)
    task.add_tags(["PyTorch-Lightning", "CNN", "SignLanguage"])
    task.connect(save_cfg)
    logger_clearml = Logger.current_logger()

    check_clearml_env()  # читаем/устанавливаем переменные среды для clearml

    dm = make_datamodule(args)  # готовим датасеты

    best_model_path = run_training(args, dm)  # обучаем модель

    if args.inference_test:
        inference_on_test(best_model_path, dm)  # инференс на всем тестовом датасете

    print("✅ Завершаю Task'у...")
    task.close()
