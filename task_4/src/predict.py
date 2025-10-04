import pandas as pd
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from transformers import ViTImageProcessor, ViTModel
from custom_classes import CFG, ViTBinaryClassifier, MedicalNetModule


def parse_args():
    parser = argparse.ArgumentParser(description='Инференс модели на каталоге изображений')

    parser.add_argument('--data_dir', type=str,
                        help='Путь к директории с изображениями для инференса')
    parser.add_argument('--weights_path', type=str,
                        help='Путь для загрузки весов модели')
    parser.add_argument('--output_csv', type=str,
                        help='Путь для сохранения результатов CSV')

    args = parser.parse_args()

    # Создаем объект конфигурации со значениями по умолчанию
    cfg = CFG()

    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    else:
        cfg.data_dir = cfg.data_dir / "test_png"

    if args.weights_path:
        cfg.weights_path = Path(args.weights_path)

    if args.output_csv:
        cfg.output_csv = Path(args.output_csv)
        cfg.output_csv.mkdir(parents=True, exist_ok=True)
    else:
        cfg.output_csv = cfg.logs_dir / cfg.output_csv

    return cfg


def process_single_image(image_path, model, image_processor, device):
    """Обработка одного изображения"""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = image_processor(image, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(inputs)
            prob = torch.sigmoid(logits).item()
            pred = int(prob > 0.5)

        return {'filename': image_path.name,
                'probability': prob,
                'prediction': pred,
                'status': 'success'
                }
    except Exception as e:
        return {'filename': image_path.name,
                'probability': None,
                'prediction': None,
                'status': f'error: {str(e)}'
                }


def process_directory(directory_path, model, image_processor, device, output_csv=None):
    """
    Обработка всех изображений в директории

    Args:
        directory_path: Путь к директории с изображениями
        model: Модель для инференса
        image_processor: Препроцессор изображений
        device: Устройство для вычислений
        output_csv: Путь для сохранения результатов (опционально)
    """
    directory_path = Path(directory_path)

    # Поддерживаемые форматы изображений
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # Находим все изображения в директории
    image_files = []
    for format in supported_formats:
        image_files.extend(directory_path.glob(f"*{format}"))
        image_files.extend(directory_path.glob(f"*{format.upper()}"))

    print(f"📁 Найдено {len(image_files)} изображений в директории: {directory_path}")

    if len(image_files) == 0:
        print("❌ Изображения не найдены. Поддерживаемые форматы:", supported_formats)
        return None

    results = []

    # Обрабатываем изображения с прогресс-баром
    for image_path in tqdm(image_files, desc="Обработка изображений"):
        result = process_single_image(image_path, model, image_processor, device)
        results.append(result)

    # Создаем DataFrame с результатами
    df_results = pd.DataFrame(results)

    # Статистика
    successful = len(df_results[df_results['status'] == 'success'])
    errors = len(df_results[df_results['status'] != 'success'])

    print(f"\n📊 Результаты обработки:")
    print(f"✅ Успешно обработано: {successful}")
    print(f"❌ Ошибок: {errors}")

    if successful > 0:
        avg_prob = df_results[df_results['status'] == 'success']['probability'].mean()
        class_0 = len(
            df_results[(df_results['status'] == 'success') & (df_results['prediction'] == 0)])
        class_1 = len(
            df_results[(df_results['status'] == 'success') & (df_results['prediction'] == 1)])

        print(f"📈 Средняя вероятность: {avg_prob:.4f}")
        print(f"🎯 Класс 0: {class_0} изображений")
        print(f"🎯 Класс 1: {class_1} изображений")

    # Сохраняем результаты в CSV если указан путь
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_path, index=False, encoding='utf-8')
        print(f"💾 Результаты сохранены в: {output_path}")

    # Показываем несколько примеров
    if successful > 0:
        print("\n📋 Примеры результатов:")
        success_results = df_results[df_results['status'] == 'success'].head(5)
        for _, row in success_results.iterrows():
            print(
                f"  {row['filename']}: prob={row['probability']:.4f}, pred={row['prediction']}")

    # Показываем ошибки если есть
    if errors > 0:
        print(f"\n⚠️  Файлы с ошибками:")
        error_results = df_results[df_results['status'] != 'success'].head(5)
        for _, row in error_results.iterrows():
            print(f"  {row['filename']}: {row['status']}")

    return df_results


if __name__ == '__main__':
    # ====== Конфигурация ======
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============== Парсим аргументы командной строки =========================
    cfg = parse_args()

    cfg.model_name = "itsomk/vit-xray-v1"
    checkpoint_path = cfg.weights_path / "best_model.ckpt"
    image_path = "Z:/python-datasets/CXR8/ct_dataset_png/pneumonia_anon_10000492_anon.png"

    # ====== Загружаем модель ======
    vit_model = ViTBinaryClassifier(pretrained_model=cfg.model_name)

    # Загружаем через LightningModule
    model = MedicalNetModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        resnet_model=vit_model,  # передаем ViT модель
        map_location=device
    )
    model.to(device)
    model.eval()

    # ====== Загружаем препроцессор ======
    image_processor = ViTImageProcessor.from_pretrained(cfg.model_name)

    # Обработка всего каталога
    image_dir = cfg.data_dir
    output_csv = cfg.output_csv

    print("🚀 Запуск инференса для каталога...")
    print(f"📁 Входная директория: {image_dir}")
    print(f"💾 Выходной файл: {output_csv}")
    print(f"⚙️  Устройство: {device}")
    print("-" * 50)

    results_df = process_directory(
        directory_path=image_dir,
        model=model,
        image_processor=image_processor,
        device=device,
        output_csv=output_csv
    )

    print("\n🎉 Инференс завершен!")
