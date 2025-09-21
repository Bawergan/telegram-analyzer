# run_importer.py

import argparse
import json
import logging
from pathlib import Path
from typing import List

# Импортируем необходимые компоненты из нашего модуля database
from database import (
    Base,
    SessionLocal,
    engine,
    settings,
    validate_and_load_from_dict
)

# Настраиваем базовое логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_export_files(root_dir: Path) -> List[Path]:
    """
    Рекурсивно находит все файлы result.json в указанной директории.
    """
    logging.info(f"Searching for 'result.json' files in '{root_dir}'...")
    files = list(root_dir.rglob("result.json"))
    logging.info(f"Found {len(files)} export file(s).")
    return files

def run_batch_import(source_dir: str):
    """
    Основная функция для запуска пакетного импорта.

    Args:
        source_dir (str): Путь к корневой директории, содержащей папки с экспортами.
    """
    root_path = Path(source_dir)
    if not root_path.is_dir():
        logging.error(f"Error: Source directory not found at '{source_dir}'")
        return

    export_files = find_export_files(root_path)
    if not export_files:
        logging.warning("No 'result.json' files found. Nothing to import.")
        return

    # Создаем таблицы в БД, если их еще не существует
    Base.metadata.create_all(bind=engine)

    # Создаем одну сессию на весь процесс для эффективности
    db = SessionLocal()
    try:
        total_files = len(export_files)
        for i, json_path in enumerate(export_files):
            export_root = json_path.parent
            logging.info(f"--- Processing file {i + 1}/{total_files}: {json_path} ---")

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # Вызываем основную логику импорта из модуля database
                chat = validate_and_load_from_dict(
                    db=db,
                    raw_data=raw_data,
                    export_root=str(export_root),
                    media_storage_root=settings.MEDIA_STORAGE_ROOT
                )

                if chat:
                    logging.info(f"Successfully processed chat '{chat.name}' (ID: {chat.telegram_id})")
                else:
                    logging.warning(f"Failed to process or validate data from {json_path}")

            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from file: {json_path}", exc_info=True)
            except Exception:
                logging.error(f"An unexpected error occurred while processing {json_path}", exc_info=True)

    finally:
        # Гарантируем закрытие сессии в любом случае
        db.close()
        logging.info("--- Batch import process finished. Database session closed. ---")


if __name__ == "__main__":
    # Настраиваем парсер аргументов командной строки
    parser = argparse.ArgumentParser(
        description="Batch importer for Telegram chat exports.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the root directory containing chat export folders.\n"
             "Example: --source /path/to/my/telegram_exports"
    )

    args = parser.parse_args()
    
    run_batch_import(args.source)