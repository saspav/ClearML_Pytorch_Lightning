import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import argparse

import warnings

import torch
import torch.nn.functional as F
import torchmetrics as tm
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch import nn
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from PIL import Image
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         DeviceStatsMonitor, OnExceptionCheckpoint,
                                         )

from transformers import ViTImageProcessor, ViTModel

from getpass import getpass
from clearml import Task, Logger

from set_all_seeds import set_all_seeds
from print_time import print_time, print_msg

# Принудительно устанавливаем Agg бэкенд - работает везде
matplotlib.use('Agg')

warnings.filterwarnings('ignore',
                        category=UserWarning,
                        message=".*Producer process has been terminated.*")

SEED = 127

# reproducibility
set_all_seeds(seed=SEED)


def check_clearml_env():
    """
    Установка переменных для clearml
    :return:
    """
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

    def load_env_file(file_path):
        """Загружает переменные из .env файла"""
        try:
            if not os.path.exists(file_path):
                return False

            print(f"📁 Загружаем переменные из файла: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Пропускаем комментарии и пустые строки
                    if not line or line.startswith('#') or '=' not in line:
                        continue

                    # Разделяем ключ и значение
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Удаляем кавычки если есть
                    if (value.startswith('"') and value.endswith('"')) or \
                            (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]

                    # Устанавливаем переменную, если она нужна и еще не установлена
                    if key in required_env_vars and os.getenv(key) is None:
                        os.environ[key] = value
                        print(f"   ✅ Загружено: {key}")

            return True
        except Exception as e:
            print(f"   ❌ Ошибка загрузки файла {file_path}: {e}")
            return False

    # Шаг 1: Пробуем загрузить из .env файлов
    env_files = [".env", os.path.expanduser("~/.clearml.env"), "clearml.env"]

    env_loaded = False
    for env_file in env_files:
        if load_env_file(env_file):
            env_loaded = True
            # После загрузки проверяем, все ли переменные установлены
            missing_after_load = [var for var in required_env_vars if os.getenv(var) is None]
            if not missing_after_load:
                break  # Все переменные загружены, выходим из цикла

    # Шаг 2: Устанавливаем значения по умолчанию для первых трех переменных
    for var in required_env_vars[:3]:
        if os.getenv(var) is None:
            os.environ[var] = env_vars[var]
            print(f"⚙️  Установлено значение по умолчанию для {var}")

    # Шаг 3: Проверяем отсутствующие переменные
    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]

    # Шаг 4: Запрашиваем у пользователя отсутствующие переменные
    if missing_vars:
        print("⚠️  Некоторые переменные среды ClearML отсутствуют.")
        for var in missing_vars:
            # Для секретных ключей используем getpass
            if "SECRET" in var or "KEY" in var:
                os.environ[var] = getpass(f"Введите значение для {var}: ")
            else:
                os.environ[var] = input(f"Введите значение для {var}: ")
        print("✅ Все переменные ClearML установлены.\n")
    else:
        if env_loaded:
            print("✅ Все переменные ClearML загружены из .env файлов.\n")
        else:
            print("✅ Все переменные ClearML уже установлены в окружении.\n")

    # Шаг 5: Сохраняем конфигурацию для будущего использования
    try:
        config_file = Path.home() / ".clearml.env"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write("# ClearML Configuration\n")
            f.write(f"# Generated on: {__import__('datetime').datetime.now()}\n")
            for var in required_env_vars:
                value = os.getenv(var)
                if value:
                    f.write(f'{var}="{value}"\n')
            f.write('CUBLAS_WORKSPACE_CONFIG=":4096:8"\n')
        print(f"💾 Конфигурация сохранена в: {config_file}")
    except Exception as e:
        print(f"💡 Не удалось сохранить конфигурацию: {e}")

    # Шаг 6: Устанавливаем CUBLAS переменную
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Шаг 7: Показываем итоговую конфигурацию (без секретов)
    print("\n📋 Итоговая конфигурация ClearML:")
    for var in required_env_vars:
        value = os.getenv(var)
        if value:
            if "SECRET" in var or "KEY" in var:
                masked_value = value[:4] + "****" + value[-4:] if len(value) > 8 else "****"
                print(f"  {var}: {masked_value}")
            else:
                print(f"  {var}: {value}")
    print(f"  CUBLAS_WORKSPACE_CONFIG: {os.getenv('CUBLAS_WORKSPACE_CONFIG')}")


# Дополнительная функция для проверки без интерактивного ввода
def verify_clearml_env():
    """
    Проверяет конфигурацию ClearML без интерактивного ввода
    """
    required_env_vars = [
        "CLEARML_WEB_HOST",
        "CLEARML_API_HOST",
        "CLEARML_FILES_HOST",
        "CLEARML_API_ACCESS_KEY",
        "CLEARML_API_SECRET_KEY"
    ]

    print("🔍 Проверка конфигурации ClearML...")
    all_set = True

    for var in required_env_vars:
        value = os.getenv(var)
        if not value:
            print(f"❌ {var}: не установлена")
            all_set = False
        else:
            if "SECRET" in var or "KEY" in var:
                masked_value = value[:4] + "****" + value[-4:] if len(value) > 8 else "****"
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"✅ {var}: {value}")

    if all_set:
        print("🎉 Все переменные ClearML корректно настроены!")
        return True
    else:
        print("💡 Запустите check_clearml_env() для настройки отсутствующих переменных")
        return False


# ===================== CFG ClearML =====================
@dataclass
class CFG:
    """
    Класс для хранения всех параметров конфигурации эксперимента.

    Атрибуты:
        project_name (str): Название проекта в ClearML.
        experiment_name (str): Название конкретного эксперимента в ClearML.
        test_size (float): Доля данных для валидации при разделении train/val.
        seed (int): Фиксированный seed для воспроизводимости.
        batch_size (int): Размер батча для обучения и валидации.
        max_epochs (int): Количество эпох обучения модели.
        num_workers (int): Число потоков для загрузки данных.
        root_dir (Path): Корневая папка.
        data_dir (Path): Корневая папка с датасетом CXR8.
        images_dir (Path): Путь к директории с изображениями.
        train_csv (str): Имя CSV-файла с тренировочной и валидационной выборкой.
        test_csv (str): Имя CSV-файла с тестовой выборкой.
        weights_path (str): Путь для сохранения/загрузки весов модели.
        logs_dir (Path): Папка для хранения логов (например, clearML).
    """
    project_name: str = "CT Project"
    experiment_name: str = "Lightning + Torchmetrics"
    model_name: str = "Default"
    seed: int = SEED
    test_size: float = 0.2
    batch_size: int = 512
    max_epochs: int = 20
    num_workers: int = 16
    num_samples: int = None

    root_dir: Path = Path("../")
    data_dir: Path = root_dir / "data"
    logs_dir: Path = root_dir / "reports"
    images_dir: Path = data_dir / "images"
    images_preprocessed: Path = data_dir / "processed"
    train_csv: str = "train_val_dataset.csv"
    test_csv: str = "test_dataset.csv"
    output_csv: str = "inference_results.csv"

    img_size: int = 224
    learning_rate: float = 1e-5

    weights_path: Path = root_dir / "models"

    def __post_init__(self):
        if self.num_workers > self.batch_size:
            self.num_workers = self.batch_size
        # создаём папку выходных файлов модели
        self.weights_path.mkdir(parents=True, exist_ok=True)
        # создаём подпапку для логов
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def parse_args():
    """
    Парсит аргументы командной строки и возвращает конфигурацию.

    Returns:
        CFG: Объект конфигурации с параметрами из командной строки или значениями по умолчанию.
    """
    parser = argparse.ArgumentParser(description='script for VIT model')

    # Пути к данным
    parser.add_argument('--data_dir', type=str,
                        help='Путь к директории с данными')
    parser.add_argument('--weights_path', type=str,
                        help='Путь для сохранения/загрузки весов модели')
    parser.add_argument('--logs_dir', type=str,
                        help='Путь для сохранения логов и результатов')

    # Гиперпараметры модели
    parser.add_argument('--batch_size', type=int,
                        help='Размер батча для обучения')
    parser.add_argument('--max_epochs', type=int,
                        help='Количество эпох обучения')
    parser.add_argument('--learning_rate', type=float,
                        help='Скорость обучения')
    parser.add_argument('--img_size', type=int,
                        help='Размер изображения')
    parser.add_argument('--test_size', type=float,
                        help='Доля данных для тестирования')

    # Дополнительные параметры
    parser.add_argument('--model_name', type=str,
                        help='Название модели')
    parser.add_argument('--num_workers', type=int,
                        help='Количество workers для DataLoader')
    parser.add_argument('--seed', type=int,
                        help='Seed для воспроизводимости')

    args = parser.parse_args()

    # Создаем объект конфигурации со значениями по умолчанию
    cfg = CFG()

    # Обновляем параметры, если они переданы в командной строке
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
        # Обновляем зависимые пути
        cfg.images_dir = cfg.data_dir / "images"
        cfg.images_preprocessed = cfg.data_dir / "processed"

    if args.weights_path:
        cfg.weights_path = Path(args.weights_path)

    if args.logs_dir:
        cfg.logs_dir = Path(args.logs_dir)

    if args.batch_size:
        cfg.batch_size = args.batch_size

    if args.max_epochs:
        cfg.max_epochs = args.max_epochs

    if args.learning_rate:
        cfg.learning_rate = args.learning_rate

    if args.img_size:
        cfg.img_size = args.img_size

    if args.test_size:
        cfg.test_size = args.test_size

    if args.model_name:
        cfg.model_name = args.model_name

    if args.num_workers:
        cfg.num_workers = args.num_workers

    if args.seed:
        cfg.seed = args.seed

    return cfg


# ===================== Dataset =====================
class CXR8Dataset(torch.utils.data.Dataset):
    """
    Кастомный Dataset для CXR8.

    Аргументы:
        df (pd.DataFrame): DataFrame с колонками ["Image Index", "target"].
        images_dir (Path): Путь к директории с изображениями.

    Методы:
        __len__: возвращает количество элементов в датасете.
        __getitem__: возвращает (изображение, метка) по индексу.
    """

    def __init__(self, df, images_dir, preprocessed_dir=None, transform=None,
                 return_image=False):
        self.df = df
        self.images_dir = Path(images_dir)
        self.preprocessed_dir = preprocessed_dir
        self.transform = transform
        self.return_image = return_image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["target"])
        img_path = self.images_dir / row["Image Index"]
        image_tensor = None

        if self.return_image:
            if self.transform:
                # print('self.transform:', self.transform)
                processed = self.transform(Image.open(img_path).convert("RGB"),
                                           return_tensors="pt")
                # Берём только pixel_values и убираем лишнюю размерность батча
                image_tensor = processed["pixel_values"].squeeze(0)  # [3, H, W]
                # print('image_tensor.shape:', image_tensor.shape)
                return image_tensor, label

            return Image.open(img_path).convert("RGB"), label

        if self.preprocessed_dir:
            pt_file = self.preprocessed_dir / (img_path.stem + ".pt")
            if pt_file.exists():
                # print(f'Читаю предобработанный файл: {img_path}')
                image_tensor = torch.load(pt_file)

        if image_tensor is None:
            image = Image.open(img_path).convert("L")  # ч/б → 1 канал

            if self.transform:
                image_tensor = self.transform(image)
                # Лениво кэшируем
                if self.preprocessed_dir is not None:
                    pt_file = self.preprocessed_dir / (img_path.stem + ".pt")
                    torch.save(image_tensor, pt_file)
            else:
                image_tensor = to_tensor(image)  # (1, H, W), float32, [0, 1]

        return image_tensor, label


# ===================== BatchAugmentorGPU =====================
class BatchAugmentorGPU:
    """
    Полностью батчевые аугментации на GPU:
    - Resize
    - Random horizontal flip
    - Random rotation ±10°
    - Random zoom ±10%
    - Random shift ±10% по каждой оси
    - Normalize
    """

    def __init__(self, img_size=None, mean=None, std=None, train=True):
        self.img_size = img_size
        self.train = train
        self.mean = mean
        self.std = std

    def __call__(self, batch: torch.Tensor, device: str = "cuda"):
        """
        batch: Tensor [B, C, H, W]
        device: куда кидать батч
        """
        batch = batch.to(device)

        if self.img_size is not None:
            # Resize
            batch = F.interpolate(batch, size=(self.img_size, self.img_size),
                                  mode='bilinear', align_corners=False)

        if self.train:
            # --- Random Flip по горизонтали ---
            flip_mask = torch.rand(batch.size(0), device=device) > 0.5
            batch[flip_mask] = batch[flip_mask].flip(dims=[3])

            # --- Random Rotation ±10 градусов ---
            angles = (torch.rand(batch.size(0), device=device) - 0.5) * 20  # градусы
            angles_rad = angles * torch.pi / 180.0
            batch = self.rotate_batch(batch, angles_rad, device)

            # --- Random Zoom (0.9–1.1) ---
            scales = 0.9 + 0.2 * torch.rand(batch.size(0), device=device)
            batch = self.zoom_batch(batch, scales, device)

        if self.mean is not None and self.std is not None:
            # Normalize
            batch = (batch - self.mean) / self.std

        return batch.cpu()

    @staticmethod
    def rotate_batch(batch, angles_rad, device):
        B, C, H, W = batch.shape
        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] = torch.cos(angles_rad)
        theta[:, 0, 1] = -torch.sin(angles_rad)
        theta[:, 1, 0] = torch.sin(angles_rad)
        theta[:, 1, 1] = torch.cos(angles_rad)
        grid = F.affine_grid(theta, batch.size(), align_corners=False)
        batch = F.grid_sample(batch, grid, mode='bilinear', padding_mode='border',
                              align_corners=False)
        return batch

    @staticmethod
    def zoom_batch(batch, scales, device):
        B, C, H, W = batch.shape
        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] = scales
        theta[:, 1, 1] = scales
        grid = F.affine_grid(theta, batch.size(), align_corners=False)
        batch = F.grid_sample(batch, grid, mode='bilinear', padding_mode='border',
                              align_corners=False)
        return batch


class CollateFn:
    """
    Функция для обработки изображений батчами для ускорения
    """

    def __init__(self, augmentor, device: str, train: bool = False):
        self.augmentor = augmentor
        self.device = device
        self.train = train
        # print(f'Инициализация аугментора: train={self.train}')

    def __call__(self, batch):
        imgs, labels = zip(*batch)

        # Сборка батча
        if getattr(self.augmentor, "image_processor_type", '') in ('YolosImageProcessor',
                                                                   'DetrImageProcessor',
                                                                   "ViTImageProcessor",
                                                                   ):
            inputs = self.augmentor(images=imgs, return_tensors="pt")
            labels = torch.tensor(labels, dtype=torch.long)
            return inputs, labels

        if getattr(self.augmentor, "image_processor_type", '') in (
                'CustomBitImageProcessor',):

            imgs = torch.stack(imgs)  # [B, C, H, W]
            labels = torch.tensor(labels)

        else:
            # ---- Случай: стандартный augmentor (старый код) ----
            imgs = torch.stack(imgs)  # [B, C, H, W]
            labels = torch.tensor(labels)
            # Аугментации (если augmentor передан)
            if self.augmentor is not None:
                # print(f'Вызов аугментора: train={self.train}')
                imgs = self.augmentor(imgs, device=self.device)

            # Для 3D-классификатора добавляем фиктивную ось глубины
            imgs = imgs.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

        return imgs, labels


# ===================== DataModule =====================
class CXR8DataModule(pl.LightningDataModule):
    def __init__(self, cfg_class, device, transformer=None, processor=None, use_images=False,
                 sample=None, mean=None, std=None, calc_stats=False):
        """
        Датамодуль
        :param cfg_class: класс конфигурации
        :param device: устройство
        :param transformer: трансформер изображения для датасета
        :param processor: процессор изображения
        :param use_images: использовать в даталоадере картинки вместо тензоров
        :param sample: количество примеров для отладки
        :param mean: среднее по датасету картинок
        :param std: стандартное отклонение по датасету картинок
        :param calc_stats: считать статистики по датасету картинок
        """
        super().__init__()
        self.cfg = cfg_class
        self.calc_stats = calc_stats
        self.sample, self.mean, self.std = sample, mean, std
        self.train_dataset = self.valid_dataset = self.test_dataset = None
        self.train_augmentor = self.valid_augmentor = None
        self.device = device
        self.transformer = transformer
        self.processor = processor  # <- добавили
        self.use_images = use_images  # <- добавили
        self.dataset_pin_memory = True
        self.class_weights = None  # <-- сюда сохраним веса классов

    def setup(self, stage=None):
        # -------------------- Загружаем CSV --------------------
        df = pd.read_csv(self.cfg.data_dir / self.cfg.train_csv)
        test_df = pd.read_csv(self.cfg.data_dir / self.cfg.test_csv)

        if self.sample is not None:
            df, test_df = train_test_split(df,
                                           train_size=self.sample,
                                           random_state=self.cfg.seed,
                                           stratify=df["target"])
            test_df = test_df.sample(self.sample // 2, random_state=self.cfg.seed)

        train_df, valid_df = train_test_split(df,
                                              test_size=self.cfg.test_size,
                                              random_state=self.cfg.seed,
                                              stratify=df["target"]
                                              )
        if self.sample is None:
            test_df = valid_df.copy()

        # ====== считаем веса ======
        class_counts_ = train_df["target"].value_counts().sort_index()

        self.class_weights = torch.tensor([class_counts_[0] / class_counts_[1]],
                                          dtype=torch.float32).to(self.device)

        # ====== метки ======
        print("Train label counts:\n", class_counts_)
        print("Train label ratio (pos fraction):", train_df["target"].mean())
        print("Class weights:", self.class_weights)

        # -------------------- Подсчёт mean/std --------------------
        if self.calc_stats and (self.mean is None or self.std is None):
            # Изменение размера картинки
            img_transform = transforms.Compose([transforms.Resize((cfg.img_size,
                                                                   cfg.img_size)),
                                                transforms.ToTensor(),
                                                ])
            tmp_dataset = CXR8Dataset(train_df, self.cfg.images_dir, transform=img_transform)
            loader = DataLoader(tmp_dataset,
                                batch_size=self.cfg.batch_size,
                                shuffle=False,
                                pin_memory=self.dataset_pin_memory,
                                num_workers=self.cfg.num_workers,
                                persistent_workers=True if self.cfg.num_workers > 0 else False,
                                )
            sum_, sum_sq, nb_samples = 0.0, 0.0, 0
            for imgs, _ in tqdm(loader, desc="Computing dataset mean/std"):
                imgs = imgs.float()
                sum_ += imgs.sum()
                sum_sq += (imgs ** 2).sum()
                nb_samples += imgs.numel()
            self.mean = (sum_ / nb_samples).item()
            self.std = torch.sqrt(sum_sq / nb_samples - self.mean ** 2).item()

        if self.mean is not None and self.std is not None:
            print(f"Dataset mean: {self.mean:.5f}, std: {self.std:.5f}")

        # Изменение размера картинки
        image_transform = transforms.Compose([
            transforms.Resize((self.cfg.img_size, self.cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.25])  # для grayscale
        ])

        preprocessed_dir = self.cfg.images_preprocessed

        preprocessed_dir = None  # кеширование картинок отключено

        if preprocessed_dir is None:
            image_transform = None
            cfg_img_size = self.cfg.img_size
        else:
            cfg_img_size = None

        if self.transformer is not None:
            image_transform = self.transformer

        # -------------------- Создаём датасеты --------------------
        self.train_dataset = CXR8Dataset(train_df, self.cfg.images_dir,
                                         preprocessed_dir=preprocessed_dir,
                                         transform=image_transform,
                                         return_image=self.use_images,
                                         )
        self.valid_dataset = CXR8Dataset(valid_df, self.cfg.images_dir,
                                         preprocessed_dir=preprocessed_dir,
                                         transform=image_transform,
                                         return_image=self.use_images,
                                         )
        self.test_dataset = CXR8Dataset(test_df, self.cfg.images_dir,
                                        preprocessed_dir=preprocessed_dir,
                                        transform=image_transform,
                                        return_image=self.use_images,
                                        )

        # ---------------------------
        # Создаем батчевый аугментор GPU
        # ---------------------------

        # -------------------- Аугменторы --------------------
        self.train_augmentor = BatchAugmentorGPU(img_size=cfg_img_size,
                                                 mean=self.mean,
                                                 std=self.std,
                                                 train=True)
        self.valid_augmentor = BatchAugmentorGPU(img_size=cfg_img_size,
                                                 mean=self.mean,
                                                 std=self.std,
                                                 train=False)

        if self.transformer is not None:
            self.train_augmentor = self.valid_augmentor = self.transformer

    # -------------------- DataLoaders --------------------
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.num_workers,
                          pin_memory=self.dataset_pin_memory,
                          persistent_workers=True if self.cfg.num_workers > 0 else False,
                          collate_fn=CollateFn(
                              self.processor if self.processor else self.train_augmentor,
                              self.device, train=True),
                          )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.cfg.batch_size,
                          shuffle=False,
                          num_workers=self.cfg.num_workers,
                          pin_memory=self.dataset_pin_memory,
                          persistent_workers=True if self.cfg.num_workers > 0 else False,
                          collate_fn=CollateFn(
                              self.processor if self.processor else self.valid_augmentor,
                              self.device, train=False),
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.cfg.batch_size,
                          shuffle=False,
                          num_workers=self.cfg.num_workers,
                          pin_memory=False,
                          persistent_workers=True if self.cfg.num_workers > 0 else False,
                          collate_fn=CollateFn(
                              self.processor if self.processor else self.valid_augmentor,
                              self.device, train=False),
                          )


class MedicalNetClassifier(nn.Module):
    """
    Модификация модели из пакета MedicalNet
    """

    def __init__(self, base_model, num_classes=2):
        super().__init__()

        # если модель в DataParallel — достаём оригинал
        if isinstance(base_model, nn.DataParallel):
            base_model = base_model.module

        self.base_model = base_model

        # убираем сегментационный head и добавляем классификацию
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )

        # классификатор
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # (B, C, 1, 1, 1)
            nn.Flatten(),
            nn.Linear(512 * base_model.layer4[0].expansion, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MedicalNetModule(pl.LightningModule):
    """
    LightningModule для бинарной классификации (здоров / болен).
    Метрики:
      - Recall (Sensitivity) — основная
      - Specificity — для контроля FP
      - AUC_ROC — общая
      - Accuracy — вспомогательная
    """

    def __init__(self, resnet_model, weight=None, learning_rate: float = 1e-4):
        super().__init__()
        self.lr = learning_rate
        self.save_hyperparameters()

        # Базовая модель (MedicalNet ResNet)
        self.model = resnet_model

        self.log_clrml = Logger.current_logger()

        # Функция потерь
        # self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

        # --- Метрики ---
        # Accuracy
        self.train_acc = tm.Accuracy(task="binary")
        self.valid_acc = tm.Accuracy(task="binary")
        # Recall (Sensitivity)
        self.train_recall = tm.Recall(task="binary")
        self.valid_recall = tm.Recall(task="binary")
        # Specificity
        self.train_spec = tm.Specificity(task="binary")
        self.valid_spec = tm.Specificity(task="binary")
        # AUC_ROC
        self.train_auc = tm.AUROC(task="binary").cpu()
        self.valid_auc = tm.AUROC(task="binary").cpu()
        # F1-score
        self.train_f1 = tm.F1Score(task="binary")
        self.valid_f1 = tm.F1Score(task="binary")

    def forward(self, x):
        """Прямой проход"""
        return self.model(x)

    def basic_step(self, batch, batch_idx, step: str):
        """
        Общий шаг для train/valid:
          - считает loss
          - обновляет метрики
          - логирует в Lightning
        """
        x, y = batch

        # y = y.float()  # 🔥

        logits = self(x)
        loss = self.criterion(logits, y.float())

        probs = torch.sigmoid(logits)  # вероятность класса 1
        preds = (probs > 0.5).long()  # бинарные предсказания

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # --- Метрики ---
            acc = getattr(self, f"{step}_acc")(preds, y)
            recall = getattr(self, f"{step}_recall")(preds, y)
            spec = getattr(self, f"{step}_spec")(preds, y)
            f1 = getattr(self, f"{step}_f1")(preds, y)
            auc = getattr(self, f"{step}_auc")(probs.cpu(), y.cpu())

        metrics_dict = {f"{step}_loss": loss,
                        f"{step}_acc": acc,
                        f"{step}_recall": recall,
                        f"{step}_spec": spec,
                        f"{step}_f1": f1,
                        f"{step}_auc": auc,
                        }

        self.log_dict(metrics_dict, prog_bar=True, on_step=False, on_epoch=True)
        # self.log_dict(metrics_dict, prog_bar=True)

        # Логирование в ClearML
        self.log_clrml.report_scalar(f"{step}_loss", "epoch", loss.item(), self.current_epoch)
        self.log_clrml.report_scalar(f"{step}_acc", "epoch", acc, self.current_epoch)
        self.log_clrml.report_scalar(f"{step}_recall", "epoch", recall, self.current_epoch)
        self.log_clrml.report_scalar(f"{step}_spec", "epoch", spec, self.current_epoch)
        self.log_clrml.report_scalar(f"{step}_f1", "epoch", f1, self.current_epoch)
        self.log_clrml.report_scalar(f"{step}_auc", "epoch", auc, self.current_epoch)

        return metrics_dict

    def training_step(self, batch, batch_idx):
        metrics_dict = self.basic_step(batch, batch_idx, "train")
        return metrics_dict["train_loss"]

    def validation_step(self, batch, batch_idx):
        metrics_dict = self.basic_step(batch, batch_idx, "valid")
        return metrics_dict["valid_loss"]

    def configure_optimizers(self):
        """Оптимизатор"""
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ===================== Training =====================
def run_training(cfg: dataclass, datamodule: pl.LightningDataModule, resnet_model,
                 task: Task, logger_clearml: Logger, use_class_weights=False,
                 monitor_metric='valid_f1', monitor_metric_mode='max', fast_dev_run=False,
                 save_last_model=True,
                 ):
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
        datamodule: экземпляр класса CXR8DataModule
    """
    weight = None
    if use_class_weights:
        weight = getattr(datamodule, 'class_weights', None)

    print('weight:', weight)

    # Инициализация модели
    model = MedicalNetModule(resnet_model, weight=weight, learning_rate=cfg.learning_rate)

    # Оптимизация умножений матриц (PyTorch 2.0+)
    try:
        torch.set_float32_matmul_precision('high')
    except:
        pass

    # Логгер Lightning (сохраняет метрики в CSV)
    csv_logger = CSVLogger(str(cfg.logs_dir), name="experiment")

    # Определение устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # ===================== Быстрый тестовый прогон =====================
    if fast_dev_run:
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
        dirpath=cfg.weights_path,
        filename="best_model",
        save_top_k=1,
        monitor=monitor_metric,  # метрика для раннего стопа
        mode=monitor_metric_mode,  # максимизируем / минимизируем метрику
        # save_weights_only=True,  # сохраняем только веса
    )
    # ранний стоп
    early_stopping_callback = EarlyStopping(
        patience=3,  # если не улучшается 3 эпохи подряд — останавливаем
        monitor=monitor_metric,  # метрика для раннего стопа
        mode=monitor_metric_mode,  # максимизируем / минимизируем метрику
        verbose=True
    )
    # логируем статистику устройств
    device_monitor_callback = DeviceStatsMonitor(cpu_stats=True)
    # чекпоинт для сохранения модели при возникновении исключения
    on_exception_callback = OnExceptionCheckpoint(dirpath=cfg.weights_path,
                                                  filename="on_exception",
                                                  )
    trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                         accelerator=accelerator,
                         log_every_n_steps=cfg.batch_size,  # Как часто логируем метрики
                         check_val_every_n_epoch=1,  # Запускаем проверку valid каждую эпоху
                         logger=csv_logger,
                         deterministic=False,  # ⚠️ ключевое исправление
                         # добавляем колбэки
                         callbacks=[checkpoint_callback,
                                    early_stopping_callback,
                                    # device_monitor_callback,
                                    on_exception_callback,
                                    ]
                         )

    start = time.time()
    trainer.fit(model, datamodule)
    print(f"Обучение завершено за {time.time() - start:.1f} секунд")

    # ===================== Сохранение весов =====================
    ckpt_path = cfg.weights_path / "model.ckpt"
    if save_last_model:
        try:
            trainer.save_checkpoint(str(ckpt_path))
            print(f"Модель сохранена: {ckpt_path}")
        except Exception as err:
            print(f'Модель НЕ сохранена, ошибка: {err}')

    # ===================== Визуализация кривых обучения =====================
    metrics = pd.read_csv(csv_logger.log_dir + "/metrics.csv")
    print('Метрики прочитаны.')

    try:
        plt.figure(figsize=(12, 5))
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(metrics["train_loss"].dropna(), label="train_loss")
        plt.plot(metrics["valid_loss"].dropna(), label="valid_loss")
        plt.legend()
        plt.title("Loss")

        # Accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(metrics["train_f1"].dropna(), label="train_f1")
        plt.plot(metrics["valid_f1"].dropna(), label="valid_f1")
        plt.legend()
        plt.title("F1")

        # Сохраняем график
        fig_path = cfg.logs_dir / "training_curves.png"
        plt.savefig(fig_path)
        plt.close()

        # ===================== Отправка графика в ClearML =====================
        logger_clearml.report_image(
            title="Learning Curves",
            series="Metrics",
            local_path=str(fig_path),
        )
    except Exception as err:
        print(f'График не построился, ошибка: {err}')

    try:
        # ===================== Инференс на одном примере =====================
        for imgs, labels in datamodule.test_dataloader():
            model.eval()
            model.to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(imgs.to(device)))  # вероятность класса 1
                preds = (probs > 0.5).long()  # бинарные предсказания
            print(f"Пример инференса: \ntrue={labels.cpu().tolist()[:13]}, "
                  f"\npred={preds.cpu().tolist()[:13]}"
                  f"\nprobs={probs.cpu().numpy().round(2).tolist()[:13]}")
            break  # берем только первый батч
    except RuntimeError as err:
        print(err)

    return str(ckpt_path)


class ViTBinaryClassifier(nn.Module):
    """
    Классификатор на основе предобученной модели vit-xray
    """

    def __init__(self, pretrained_model="itsomk/vit-xray-v1",
                 freeze_backbone=True):
        super().__init__()

        # Загружаем backbone без головы классификации
        self.vit = ViTModel.from_pretrained(pretrained_model)

        hidden_dim = self.vit.config.hidden_size  # обычно 768 для base

        # Классификатор поверх CLS-токена
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)  # один логит вместо 2
        )

        # Замораживаем backbone (если хотим дообучать только голову)
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            # Размораживаем только pooler
            for pool_param in self.vit.pooler.parameters():
                pool_param.requires_grad = True

    def forward(self, inputs):
        # inputs = {"pixel_values": tensor[B, 3, 224, 224]}
        outputs = self.vit(**inputs)
        hidden_states = outputs.last_hidden_state  # (B, num_patches+1, hidden_dim)

        # CLS-токен
        cls_token = hidden_states[:, 0, :]  # (B, hidden_dim)

        # Классификация
        logits = self.classifier(cls_token)  # (B, 1)
        return logits.squeeze(-1)  # (B,) для совместимости с BCEWithLogitsLoss


if __name__ == '__main__':

    # =============== Проверка переменных для clearml ==========================
    check_clearml_env()

    # ============== Парсим аргументы командной строки =========================
    cfg = parse_args()
    # Выводим текущую конфигурацию
    print("📋 Текущая конфигурация:")
    for field_name, field_value in cfg.__dict__.items():
        print(f"  {field_name}: {field_value}")

    # ============== Проверка датамодуля и даталоадера =========================

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', type(device_), device_)

    cfg_ = CFG()

    cfg_.batch_size = 256
    cfg_.num_workers = 24
    cfg_.test_size = 0.05

    datamodule_time = print_msg('Формирование датамодуля...')
    datamodule_ = CXR8DataModule(cfg_, 'cpu', calc_stats=True)
    datamodule_.setup()  # автоматически считает mean/std и создаёт датасеты
    print_time(datamodule_time)

    # Mean/std dataset
    print("Train dataset mean/std:", datamodule_.mean, datamodule_.std)

    check_time = print_msg("Проверка одного батча train'a...")
    for _imgs, _labels in datamodule_.train_dataloader():
        print("Batch images shape:", _imgs.shape)  # [B, C, H, W]
        print("Batch labels shape:", _labels.shape)  # [B]
        print("dtype:", _imgs.dtype, _labels.dtype)  # torch.float32 / torch.int64
        print("device:", _imgs.device, _labels.device)  # должно быть cuda
        print("Batch images min/max:", _imgs.min().item(), _imgs.max().item())
        print("Пример меток:", _labels[:10].tolist())  # первые 10
        break
    print_time(check_time)
