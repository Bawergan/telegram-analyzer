# test_run_importer.py

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Импортируем функции, которые будем тестировать
from run_importer import find_export_files, run_batch_import

# --- Тесты для вспомогательной функции find_export_files ---

def test_find_export_files_successfully(tmp_path: Path):
    """
    Проверяет, что функция корректно находит result.json в сложной структуре.
    """
    (tmp_path / "chat1").mkdir()
    (tmp_path / "chat1" / "result.json").touch()
    (tmp_path / "empty_dir").mkdir()
    (tmp_path / "deep" / "nested" / "chat2").mkdir(parents=True)
    (tmp_path / "deep" / "nested" / "chat2" / "result.json").touch()
    (tmp_path / "not_a_result.txt").touch()

    found_files = find_export_files(tmp_path)

    assert len(found_files) == 2
    assert set(found_files) == {
        tmp_path / "chat1" / "result.json",
        tmp_path / "deep" / "nested" / "chat2" / "result.json"
    }

def test_find_export_files_no_files_found(tmp_path: Path):
    """
    Проверяет, что функция возвращает пустой список, если нет result.json.
    """
    (tmp_path / "chat1").mkdir()
    (tmp_path / "chat1" / "some_other_file.txt").touch()

    found_files = find_export_files(tmp_path)
    assert not found_files

# --- Тесты для основной функции run_batch_import ---

@patch('run_importer.validate_and_load_from_dict')
@patch('run_importer.SessionLocal')
@patch('run_importer.Base.metadata.create_all')
@patch('run_importer.find_export_files')
@patch('run_importer.settings')
def test_run_batch_import_happy_path(
    mock_settings: MagicMock,
    mock_find_files: MagicMock,
    mock_create_all: MagicMock,
    mock_session_local: MagicMock,
    mock_validate: MagicMock,
    tmp_path: Path,
    caplog
):
    """
    Тестирует основной успешный сценарий работы batch-импортера.
    """
    caplog.set_level(logging.INFO)

    # 1. Настройка mock-объектов
    mock_session_instance = MagicMock()
    mock_session_local.return_value = mock_session_instance
    
    file1_path = tmp_path / "chat1" / "result.json"
    file2_path = tmp_path / "chat2" / "result.json"
    mock_find_files.return_value = [file1_path, file2_path]
    
    mock_chat_object = MagicMock()
    mock_chat_object.configure_mock(
        name='Test Chat',
        telegram_id=123
    )
    # ИЗМЕНЕНИЕ: validate_and_load_from_dict теперь возвращает кортеж (chat, new_messages_count)
    mock_validate.return_value = (mock_chat_object, 10)

    mock_settings.MEDIA_STORAGE_ROOT = str(tmp_path / "media")
    
    test_json_data = {"id": 123, "name": "Test Chat", "messages": []}

    # patch Path.mkdir to avoid actual directory creation
    with patch('pathlib.Path.mkdir'):
        with patch('builtins.open', mock_open(read_data=json.dumps(test_json_data))):
            run_batch_import(str(tmp_path))

    # 3. Проверки (Asserts)
    mock_create_all.assert_called_once()
    mock_find_files.assert_called_once_with(tmp_path)
    mock_session_local.assert_called_once()
    mock_session_instance.close.assert_called_once()

    assert mock_validate.call_count == 2

    first_call_args = mock_validate.call_args_list[0].kwargs
    assert first_call_args['db'] == mock_session_instance
    assert first_call_args['raw_data'] == test_json_data
    assert first_call_args['export_root'] == str(file1_path.parent)
    
    second_call_args = mock_validate.call_args_list[1].kwargs
    assert second_call_args['export_root'] == str(file2_path.parent)

    assert "Successfully processed chat 'Test Chat' (ID: 123)" in caplog.text
    # ИСПРАВЛЕНИЕ: Эта проверка теперь должна работать, так как логи захватываются корректно
    assert "Total new messages added to the database: 20" in caplog.text


@patch('run_importer.validate_and_load_from_dict')
@patch('run_importer.find_export_files')
def test_run_batch_import_no_files_found(
    mock_find_files: MagicMock,
    mock_validate: MagicMock,
    tmp_path: Path,
    caplog
):
    """
    Проверяет, что импорт не запускается, если не найдено файлов.
    """
    mock_find_files.return_value = []
    
    with caplog.at_level(logging.WARNING):
        run_batch_import(str(tmp_path))
    
    mock_validate.assert_not_called()
    assert "No 'result.json' files found. Nothing to import." in caplog.text


@patch('run_importer.validate_and_load_from_dict')
@patch('run_importer.SessionLocal')
@patch('run_importer.Base.metadata.create_all')
@patch('run_importer.find_export_files')
@patch('run_importer.json.load') 
def test_run_batch_import_handles_json_error(
    mock_json_load: MagicMock,
    mock_find_files: MagicMock,
    mock_create_all: MagicMock,
    mock_session_local: MagicMock,
    mock_validate: MagicMock,
    tmp_path: Path,
    caplog
):
    """
    Проверяет, что скрипт обрабатывает ошибку парсинга JSON и продолжает работу.
    """
    # ИСПРАВЛЕНИЕ: Устанавливаем уровень INFO, чтобы поймать и ошибку, и финальную статистику
    caplog.set_level(logging.INFO)
    
    mock_session_instance = MagicMock()
    mock_session_local.return_value = mock_session_instance
    
    file_path = tmp_path / "bad_chat" / "result.json"
    mock_find_files.return_value = [file_path]
    
    mock_json_load.side_effect = json.JSONDecodeError("Mocked Decode Error", "doc", 0)

    with patch('pathlib.Path.mkdir'):
        with patch('builtins.open', mock_open(read_data="this is not a valid json")):
            # ИСПРАВЛЕНИЕ: Убираем with caplog.at_level(logging.ERROR), так как уровень уже задан
            run_batch_import(str(tmp_path))

    mock_validate.assert_not_called() 
    mock_session_instance.close.assert_called_once() 
    mock_session_instance.rollback.assert_called_once() 
    assert f"Error decoding JSON from file: {file_path}" in caplog.text
    # ИСПРАВЛЕНИЕ: Эта проверка теперь должна работать, так как итоговая статистика будет в логах
    assert "Failed or skipped: 1" in caplog.text


def test_run_batch_import_source_dir_not_found(tmp_path: Path, caplog):
    """
    Проверяет, что скрипт логирует ошибку, если исходная директория не найдена.
    """
    non_existent_dir = tmp_path / "non_existent"
    
    with caplog.at_level(logging.ERROR):
        run_batch_import(str(non_existent_dir))
    
    assert f"Source directory not found at '{non_existent_dir}'" in caplog.text