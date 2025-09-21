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
    Рекурсивно находит все файлы 'result.json' в указанной директории.

    Args:
        root_dir: Корневая директория для поиска.

    Returns:
        Список объектов Path, указывающих на найденные файлы 'result.json'.
    """
    logging.info(f"Searching for 'result.json' files in '{root_dir}'...")
    files = list(root_dir.rglob("result.json"))
    logging.info(f"Found {len(files)} export file(s).")
    return files

def run_batch_import(source_dir: str):
    """
    Основная функция для запуска пакетного импорта данных из экспортов Telegram.

    Функция сканирует указанную директорию, находит все файлы 'result.json',
    и для каждого из них запускает процесс валидации и загрузки данных в базу данных.
    В конце выводится итоговая статистика по импорту.

    Args:
        source_dir: Путь к корневой директории, содержащей папки с экспортами.
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
    logging.info("Initializing database and creating tables if they don't exist...")
    Base.metadata.create_all(bind=engine)

    # Улучшение: Создаем директорию для медиа один раз в начале
    media_storage_path = Path(settings.MEDIA_STORAGE_ROOT)
    media_storage_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Media storage directory is ready at: '{media_storage_path}'")

    # Инициализация статистики
    successful_files = 0
    failed_files = 0
    total_new_messages = 0

    # Создаем одну сессию на весь процесс для эффективности
    db = SessionLocal()
    try:
        total_files = len(export_files)
        logging.info(f"--- Starting batch import of {total_files} file(s) ---")

        for i, json_path in enumerate(export_files):
            export_root = json_path.parent
            logging.info(f"--- Processing file {i + 1}/{total_files}: {json_path} ---")

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # Вызываем основную логику импорта из модуля database
                chat, new_messages_count = validate_and_load_from_dict(
                    db=db,
                    raw_data=raw_data,
                    export_root=str(export_root),
                    media_storage_root=settings.MEDIA_STORAGE_ROOT
                )

                if chat:
                    logging.info(f"Successfully processed chat '{chat.name}' (ID: {chat.telegram_id}). Added {new_messages_count} new messages.")
                    successful_files += 1
                    total_new_messages += new_messages_count
                else:
                    # Ошибка валидации Pydantic, уже залогирована внутри функции
                    logging.warning(f"Failed to process or validate data from {json_path}. Rolling back transaction for this file.")
                    db.rollback()
                    failed_files += 1

            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from file: {json_path}. Skipping this file.", exc_info=True)
                db.rollback()
                failed_files += 1
            except Exception:
                # Отлавливаем другие неожиданные ошибки (например, I/O)
                logging.error(f"An unexpected error occurred while processing {json_path}. Rolling back and skipping.", exc_info=True)
                db.rollback()
                failed_files += 1

    finally:
        # Гарантируем закрытие сессии в любом случае
        db.close()
        logging.info("--- Batch import process finished. Database session closed. ---")
        
        # Вывод итоговой статистики
        logging.info("--- Import Summary ---")
        logging.info(f"Total files processed: {successful_files + failed_files}")
        # ИСПРАВЛЕНИЕ: Убраны ANSI-коды для совместимости с тестами и системами логирования
        logging.info(f"  Successfully imported: {successful_files}")
        logging.info(f"  Failed or skipped: {failed_files}")
        logging.info(f"Total new messages added to the database: {total_new_messages}")
        logging.info("----------------------")


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