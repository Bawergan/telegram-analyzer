# test_database.py

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError

# Импортируем все необходимое из нашего модуля database
from database import (
    Base,
    Message,
    User,
    Chat,
    validate_and_load_from_dict
)

# --- Фикстура для создания временной сессии БД для каждого теста ---

@pytest.fixture(scope="function")
def db_session():
    """Создает чистую базу данных в памяти для одного тестового запуска."""
    engine_test = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine_test)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine_test)

# --- Класс с набором тестов для database.py ---

class TestComprehensiveAndOptimizedDatabase:
    # --- Группа 1: Базовая логика и целостность данных ---
    def test_successful_load(self, db_session: Session, tmp_path: Path):
        sample_json = {"id": 101, "name": "T", "messages": [{"id": 1, "type": "message", "from": "A", "from_id": "u1", "text": "Hi"}]}
        validate_and_load_from_dict(db_session, sample_json, str(tmp_path), str(tmp_path))
        assert db_session.query(Message).count() == 1
        assert db_session.query(User).count() == 1
    
    def test_idempotency_by_uniqueness_constraint(self, db_session: Session):
        chat1 = Chat(telegram_id=1, name="Constraint Test")
        db_session.add(chat1)
        db_session.commit()
        db_session.add(Message(telegram_id=100, chat_id=chat1.id))
        db_session.commit()
        db_session.add(Message(telegram_id=100, chat_id=chat1.id))
        with pytest.raises(IntegrityError): db_session.commit()

    def test_foreign_key_integrity(self, db_session: Session, tmp_path: Path):
        sample_json = {"id": 404, "messages": [{"id": 30, "type": "message", "from": "T", "from_id": "ut", "text": ""}]}
        validate_and_load_from_dict(db_session, sample_json, str(tmp_path), str(tmp_path))
        message = db_session.query(Message).one()
        assert message.chat.telegram_id == 404 and message.author.name == "T"

    def test_cascade_delete(self, db_session: Session, tmp_path: Path):
        sample_json = {"id": 505, "messages": [{"id": 40, "type": "message", "from_id": "ud", "from": "D", "text": ""}]}
        chat, _ = validate_and_load_from_dict(db_session, sample_json, str(tmp_path), str(tmp_path))
        db_session.delete(chat)
        db_session.commit()
        assert db_session.query(Message).count() == 0

    def test_unicode_handling(self, db_session: Session, tmp_path: Path):
        unicode_json = {"id": 606, "messages": [{"id": 50, "type": "message", "from": "Мария", "from_id": "um", "text": "Привет! 🌍"}]}
        validate_and_load_from_dict(db_session, unicode_json, str(tmp_path), str(tmp_path))
        assert db_session.query(Message).one().text == "Привет! 🌍" and db_session.query(User).one().name == "Мария"

    def test_alias_for_from_field(self, db_session: Session, tmp_path: Path):
        json_with_from = {"id": 303, "messages": [{"id": 20, "type": "message", "from": "Special", "from_id": "us", "text": ""}]}
        validate_and_load_from_dict(db_session, json_with_from, str(tmp_path), str(tmp_path))
        assert db_session.query(User).filter_by(telegram_id="us").one().name == "Special"
    
    def test_service_message_handling(self, db_session: Session, tmp_path: Path):
        service_msg_json = {"id": 808, "messages": [{"id": 1, "type": "service"}, {"id": 2, "type": "message", "from_id": "u1", "from": "U"}]}
        validate_and_load_from_dict(db_session, service_msg_json, str(tmp_path), str(tmp_path))
        msg1, msg2 = db_session.query(Message).order_by(Message.telegram_id).all()
        assert msg1.is_service and msg1.author_id is None and not msg2.is_service and msg2.author_id is not None
    
    def test_idempotency_skips_unchanged_messages(self, db_session: Session, tmp_path: Path):
        sample_json = {"id": 707, "messages": [{"id": i, "type": "message", "from_id": f"u{i % 2}", "from": "User"} for i in range(10)]}
        _, count1 = validate_and_load_from_dict(db_session, sample_json, str(tmp_path), str(tmp_path))
        assert count1 == 10
        assert db_session.query(Message).count() == 10
        _, count2 = validate_and_load_from_dict(db_session, sample_json, str(tmp_path), str(tmp_path))
        assert count2 == 0 # Новых сообщений не должно быть добавлено
        assert db_session.query(Message).count() == 10

    def test_raw_text_storage(self, db_session: Session, tmp_path: Path):
        structured_text = ["Link test: ", {"type": "link", "text": "https://example.com"}]
        json_data = {"id": 1010, "messages": [{"id": 1, "type": "message", "from_id": "u1", "from": "U", "text": structured_text}]}
        validate_and_load_from_dict(db_session, json_data, str(tmp_path), str(tmp_path))
        msg = db_session.query(Message).one()
        assert msg.text == "Link test: https://example.com"
        assert isinstance(msg.raw_text, list) and msg.raw_text[1]['type'] == 'link'

    # --- Группа 2: Полнота данных и граничные случаи ETL ---
    def test_full_message_data_load(self, db_session: Session, tmp_path: Path):
        full_message_json = {
            "id": 909, "name": "Full Data Chat", "type": "personal_chat",
            "messages": [
                {"id": 1, "type": "message", "text": "Это первое сообщение"},
                {
                    "id": 2, "type": "message", "from": "Tester", "from_id": "user_tester",
                    "reply_to_message_id": 1,
                    "file": "photos/photo.jpg", "media_type": "voice_message", "mime_type": "audio/ogg", "duration_seconds": 5,
                    "text": ["Это ", {"type": "bold", "text": "жирный"}, " текст"],
                    "reactions": [{"emoji": "👍", "count": 2}]
                }
            ]
        }
        validate_and_load_from_dict(db_session, full_message_json, str(tmp_path), str(tmp_path))
        assert db_session.query(Message).count() == 2
        msg = db_session.query(Message).filter_by(telegram_id=2).one()
        assert msg.reply_to_message_id == 1
        assert msg.file_path == "photos/photo.jpg"
        assert msg.media_type == "voice_message"
        assert msg.mime_type == "audio/ogg"
        assert msg.duration_seconds == 5
        assert len(msg.reactions) == 1 and msg.reactions[0].emoji == "👍"
        assert len(msg.text_entities) == 1 and msg.text_entities[0].type == "bold"
        assert msg.text == "Это жирный текст"

    def test_batch_loading_logic(self, db_session: Session, tmp_path: Path):
        batch_test_json = {
            "id": 111, "messages": [{"id": i, "type": "message", "from_id": "u1", "from": "U1"} for i in range(12)]
        }
        with patch.object(db_session, 'add_all', wraps=db_session.add_all) as mock_add_all:
            validate_and_load_from_dict(db_session, batch_test_json, str(tmp_path), str(tmp_path), batch_size=5)
            assert db_session.query(Message).count() == 12 and db_session.query(User).count() == 1
            assert mock_add_all.call_count == 4

    def test_loading_empty_messages_list(self, db_session: Session, tmp_path: Path):
        empty_json = {"id": 222, "name": "Empty Chat", "messages": []}
        chat, _ = validate_and_load_from_dict(db_session, empty_json, str(tmp_path), str(tmp_path))
        assert chat is not None and chat.telegram_id == 222
        assert db_session.query(Message).count() == 0 and db_session.query(User).count() == 0

    @patch('database.logging.error')
    def test_invalid_json_validation_fails_gracefully(self, mock_logging_error: MagicMock, db_session: Session, tmp_path: Path):
        invalid_json = {"id": 444, "name": "Invalid Chat"}
        result, count = validate_and_load_from_dict(db_session, invalid_json, str(tmp_path), str(tmp_path))
        assert result is None and count == 0 and db_session.query(Chat).count() == 0
        mock_logging_error.assert_called_once()
    
    # --- Группа 3: Upsert-логика ---
    def test_upsert_logic_updates_edited_message(self, db_session: Session, tmp_path: Path):
        chat_id = 123
        msg_id = 1
        original_text = "Original text"
        edited_text = "This text has been edited."
        edit_date = "2025-01-01T12:00:00"
        initial_json = {"id": chat_id, "messages": [{"id": msg_id, "type": "message", "text": original_text}]}
        validate_and_load_from_dict(db_session, initial_json, str(tmp_path), str(tmp_path))
        msg_from_db = db_session.query(Message).filter_by(telegram_id=msg_id).one()
        assert msg_from_db.text == original_text and msg_from_db.edited_date is None
        
        updated_json = {"id": chat_id, "messages": [{"id": msg_id, "type": "message", "text": edited_text, "edited": edit_date}]}
        validate_and_load_from_dict(db_session, updated_json, str(tmp_path), str(tmp_path))
        
        assert db_session.query(Message).count() == 1
        msg_from_db = db_session.query(Message).filter_by(telegram_id=msg_id).one()
        assert msg_from_db.text == edited_text
        assert msg_from_db.edited_date == datetime.fromisoformat(edit_date)

    def test_upsert_logic_merges_new_and_skips_existing(self, db_session: Session, tmp_path: Path):
        chat_id = 456
        initial_json = {"id": chat_id, "messages": [{"id": i, "type": "message", "text": f"Msg {i}"} for i in range(1, 11)]}
        _, count1 = validate_and_load_from_dict(db_session, initial_json, str(tmp_path), str(tmp_path))
        assert db_session.query(Message).count() == 10 and count1 == 10
        
        overlapping_json = {"id": chat_id, "messages": [{"id": i, "type": "message", "text": f"Msg {i}"} for i in range(5, 16)]}
        _, count2 = validate_and_load_from_dict(db_session, overlapping_json, str(tmp_path), str(tmp_path))
        
        assert db_session.query(Message).count() == 15 and count2 == 5
        all_ids = {msg.telegram_id for msg in db_session.query(Message).all()}
        assert all_ids == set(range(1, 16))

    def test_upsert_does_not_update_with_older_or_same_edit_date(self, db_session: Session, tmp_path: Path):
        chat_id = 789
        msg_id = 1
        newer_text = "Newer edit"
        older_text = "Older edit"
        newer_date_str = "2025-02-01T00:00:00"
        older_date_str = "2025-01-01T00:00:00"
        initial_json = {"id": chat_id, "messages": [{"id": msg_id, "type": "message", "text": newer_text, "edited": newer_date_str}]}
        validate_and_load_from_dict(db_session, initial_json, str(tmp_path), str(tmp_path))
        msg_from_db = db_session.query(Message).one()
        assert msg_from_db.text == newer_text
        
        validate_and_load_from_dict(db_session, initial_json, str(tmp_path), str(tmp_path))
        older_json = {"id": chat_id, "messages": [{"id": msg_id, "type": "message", "text": older_text, "edited": older_date_str}]}
        validate_and_load_from_dict(db_session, older_json, str(tmp_path), str(tmp_path))
        
        msg_from_db = db_session.query(Message).one()
        assert msg_from_db.text == newer_text
        assert msg_from_db.edited_date == datetime.fromisoformat(newer_date_str)
        
    # --- Группа 4: Управление медиафайлами ---

    def test_media_file_copy_and_path_storage(self, db_session: Session, tmp_path: Path):
        """Проверяет, что медиафайл корректно копируется и путь сохраняется в БД."""
        export_root = tmp_path / "export_data"
        media_storage_root = tmp_path / "central_storage"
        export_root.mkdir()
        media_storage_root.mkdir() # В реальном коде это делает run_importer
        (export_root / "photos").mkdir()
        
        dummy_file_rel_path = "photos/my_photo.jpg"
        dummy_file_abs_path = export_root / dummy_file_rel_path
        dummy_file_abs_path.write_text("image data")

        chat_id, msg_id = 12345, 67890
        json_data = {
            "id": chat_id, "messages": [
                {"id": msg_id, "type": "message", "file": dummy_file_rel_path, "media_type": "photo"}
            ]
        }
        
        validate_and_load_from_dict(db_session, json_data, str(export_root), str(media_storage_root))

        msg_from_db = db_session.query(Message).one()
        expected_new_filename = f"{chat_id}_{msg_id}.jpg"
        expected_dest_path = media_storage_root / expected_new_filename

        assert msg_from_db.file_path == dummy_file_rel_path
        assert msg_from_db.stored_media_path == expected_new_filename
        assert expected_dest_path.exists()
        assert expected_dest_path.read_text() == "image data"

    @patch('database.logging.warning')
    def test_graceful_fail_on_missing_media_file(self, mock_logging_warning: MagicMock, db_session: Session, tmp_path: Path):
        """Проверяет, что импорт не падает, если медиафайл отсутствует."""
        export_root = tmp_path / "export_data"
        media_storage_root = tmp_path / "central_storage"
        export_root.mkdir()
        media_storage_root.mkdir()

        json_data = {
            "id": 111, "messages": [
                {"id": 222, "type": "message", "file": "photos/missing.jpg", "media_type": "photo"}
            ]
        }

        validate_and_load_from_dict(db_session, json_data, str(export_root), str(media_storage_root))

        assert db_session.query(Message).count() == 1
        msg_from_db = db_session.query(Message).one()
        assert msg_from_db.stored_media_path is None
        mock_logging_warning.assert_called_once()

    def test_chat_metadata_updates_on_reimport(self, db_session: Session, tmp_path: Path):
        """Проверяет, что имя и тип чата обновляются при повторном импорте."""
        chat_id = 999
        initial_json = {"id": chat_id, "name": "Project A", "type": "private_supergroup", "messages": []}
        validate_and_load_from_dict(db_session, initial_json, str(tmp_path), str(tmp_path))
        
        chat_from_db = db_session.query(Chat).filter_by(telegram_id=chat_id).one()
        assert chat_from_db.name == "Project A" and chat_from_db.type == "private_supergroup"
        assert db_session.query(Chat).count() == 1

        updated_json = {"id": chat_id, "name": "Project B", "type": "public_supergroup", "messages": []}
        validate_and_load_from_dict(db_session, updated_json, str(tmp_path), str(tmp_path))

        assert db_session.query(Chat).count() == 1
        chat_from_db_updated = db_session.query(Chat).filter_by(telegram_id=chat_id).one()
        assert chat_from_db_updated.name == "Project B" and chat_from_db_updated.type == "public_supergroup"

    def test_user_name_updates_on_reimport(self, db_session: Session, tmp_path: Path):
        """Проверяет, что имя пользователя обновляется при повторном импорте."""
        chat_id = 888
        user_id = "user123"

        initial_json = {"id": chat_id, "messages": [{"id": 1, "type": "message", "from": "OldName", "from_id": user_id}]}
        validate_and_load_from_dict(db_session, initial_json, str(tmp_path), str(tmp_path))
        assert db_session.query(User).filter_by(telegram_id=user_id).one().name == "OldName"
        assert db_session.query(User).count() == 1

        updated_json = {"id": chat_id, "messages": [{"id": 2, "type": "message", "from": "NewName", "from_id": user_id}]}
        validate_and_load_from_dict(db_session, updated_json, str(tmp_path), str(tmp_path))

        assert db_session.query(User).count() == 1
        assert db_session.query(User).filter_by(telegram_id=user_id).one().name == "NewName"
        assert db_session.query(Message).count() == 2