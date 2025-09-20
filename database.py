# database.py

# --- 0. Импорты и конфигурация ---
import logging
import os
import sys
import json
from datetime import datetime
from typing import List, Optional, Union, Any, Dict, TypeVar, Tuple, Type

from sqlalchemy.types import UserDefinedType

import pytest
from unittest.mock import MagicMock, patch

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field, ValidationError

from sqlalchemy import (
    create_engine, String, DateTime, Text, ForeignKey,
    UniqueConstraint, JSON, Integer, Boolean
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    sessionmaker, Session, DeclarativeBase, Mapped,
    mapped_column, relationship
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Управление конфигурацией ---
if 'pytest' in sys.modules:
    os.environ['DATABASE_URL'] = 'sqlite:///:memory:'

class Settings(BaseSettings):
    DATABASE_URL: str
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')
settings = Settings()

# --- 2. Схемы Pydantic ---
class ReactionSchema(BaseModel): emoji: Optional[str] = None; count: Optional[int] = None
class TextEntitySchema(BaseModel): type: str; text: str

class MessageSchema(BaseModel):
    id: int
    type: str
    date: Optional[datetime] = None
    from_name: Optional[str] = Field(None, alias='from')
    from_id: Optional[str] = None
    text: Optional[Union[str, List[Union[str, Dict[str, Any]]]]] = None
    reply_to_message_id: Optional[int] = None
    reactions: Optional[List[ReactionSchema]] = []
    text_entities: Optional[List[TextEntitySchema]] = []
    file: Optional[str] = None
    media_type: Optional[str] = None
    mime_type: Optional[str] = None
    duration_seconds: Optional[int] = None
    model_config = {"populate_by_name": True, "extra": "ignore"}

class TelegramChatExportSchema(BaseModel):
    id: int
    name: Optional[str] = None
    type: Optional[str] = None
    messages: List[MessageSchema]

# --- 3. Конфигурация БД ---
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
class Base(DeclarativeBase): pass
T = TypeVar("T", bound=Base)

# --- 4. ORM-модели ---
class Vector(UserDefinedType):
    def get_col_spec(self, **kw): return "VECTOR"
    def bind_processor(self, dialect): return lambda v: json.dumps(v) if v and dialect.name != 'postgresql' else v
    def result_processor(self, dialect, coltype): return lambda v: json.loads(v) if v and dialect.name != 'postgresql' else v

class Chat(Base):
    __tablename__ = "chats"
    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[int] = mapped_column(unique=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    messages: Mapped[List["Message"]] = relationship("Message", back_populates="chat", cascade="all, delete-orphan")

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    messages: Mapped[List["Message"]] = relationship("Message", back_populates="author")

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[int] = mapped_column(index=True)
    date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_text: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    is_service: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id"))
    author_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    reply_to_message_id: Mapped[Optional[int]] = mapped_column(nullable=True)
    file_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    media_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    embedding = mapped_column(Vector, nullable=True)
    
    chat: Mapped["Chat"] = relationship(back_populates="messages")
    author: Mapped[Optional["User"]] = relationship(back_populates="messages")
    text_entities: Mapped[List["TextEntity"]] = relationship(back_populates="message", cascade="all, delete-orphan")
    reactions: Mapped[List["Reaction"]] = relationship(back_populates="message", cascade="all, delete-orphan")
    __table_args__ = (UniqueConstraint('telegram_id', 'chat_id', name='_telegram_id_chat_id_uc'),)

class TextEntity(Base):
    __tablename__ = "text_entities"
    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(String)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"))
    message: Mapped["Message"] = relationship(back_populates="text_entities")

class Reaction(Base):
    __tablename__ = "reactions"
    id: Mapped[int] = mapped_column(primary_key=True)
    emoji: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"))
    message: Mapped["Message"] = relationship(back_populates="reactions")


# --- 5. Интерфейс базы данных и логика ETL ---
def get_or_create(session: Session, model: Type[T], defaults: Optional[dict] = None, **kwargs) -> Tuple[T, bool]:
    instance = session.query(model).filter_by(**kwargs).one_or_none()
    if instance: return instance, False
    instance = model(**kwargs, **(defaults or {}))
    session.add(instance)
    return instance, True

def load_validated_data(db: Session, chat_data: TelegramChatExportSchema, batch_size: int = 1000):
    chat, _ = get_or_create(db, Chat, telegram_id=chat_data.id, defaults={'name': chat_data.name, 'type': chat_data.type})
    db.flush()

    existing_message_ids = {id[0] for id in db.query(Message.telegram_id).filter_by(chat_id=chat.id)}
    user_cache = {user.telegram_id: user for user in db.query(User)}
    
    messages_to_process = []
    users_to_create = {}

    for msg in chat_data.messages:
        if msg.id in existing_message_ids: continue

        if not msg.text_entities and isinstance(msg.text, list):
            try:
                entities_from_text = [item for item in msg.text if isinstance(item, dict)]
                msg.text_entities = [TextEntitySchema.model_validate(e) for e in entities_from_text]
            except ValidationError:
                logging.warning(f"Could not parse text_entities from structured text for message {msg.id}")
                msg.text_entities = []

        author_telegram_id = msg.from_id
        if author_telegram_id and author_telegram_id not in user_cache and author_telegram_id not in users_to_create:
            users_to_create[author_telegram_id] = User(telegram_id=author_telegram_id, name=msg.from_name)

        raw_text_content = msg.text
        if isinstance(raw_text_content, list):
            text_content = "".join([item.get('text', str(item)) if isinstance(item, dict) else str(item) for item in raw_text_content])
        else:
            text_content = str(raw_text_content) if raw_text_content else ""
        
        new_message = Message(
            telegram_id=msg.id, date=msg.date, text=text_content, raw_text=msg.model_dump().get('text'),
            is_service=(msg.type == 'service'), chat_id=chat.id, author_id=None,
            reply_to_message_id=msg.reply_to_message_id, file_path=msg.file, media_type=msg.media_type,
            mime_type=msg.mime_type, duration_seconds=msg.duration_seconds
        )
        new_message.text_entities = [TextEntity(type=e.type, text=e.text) for e in msg.text_entities]
        new_message.reactions = [Reaction(emoji=r.emoji, count=r.count) for r in msg.reactions]
        
        messages_to_process.append((new_message, author_telegram_id))

    if users_to_create:
        db.add_all(users_to_create.values())
        db.flush()
        for user in users_to_create.values():
            user_cache[user.telegram_id] = user

    messages_to_add = []
    for message, author_telegram_id in messages_to_process:
        if author_telegram_id and author_telegram_id in user_cache:
            message.author_id = user_cache[author_telegram_id].id
        
        messages_to_add.append(message)
        if len(messages_to_add) >= batch_size:
            db.add_all(messages_to_add)
            messages_to_add = []
            
    if messages_to_add:
        db.add_all(messages_to_add)

    db.commit()
    return chat

def validate_and_load_from_dict(db: Session, raw_data: dict, batch_size: int = 1000):
    try:
        validated_chat_export = TelegramChatExportSchema.model_validate(raw_data)
        return load_validated_data(db, validated_chat_export, batch_size)
    except ValidationError:
        logging.error("JSON validation failed.", exc_info=True)
        return None

# --- 6. Модульное тестирование ---
@pytest.fixture(scope="function")
def db_session():
    engine_test = create_engine(settings.DATABASE_URL)
    Base.metadata.create_all(engine_test)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)
    db = TestSessionLocal()
    yield db
    db.close()
    Base.metadata.drop_all(engine_test)

class TestComprehensiveAndOptimizedDatabase:
    # --- Группа 1: Базовая логика и целостность данных ---
    def test_successful_load(self, db_session: Session):
        sample_json = {"id": 101, "name": "T", "messages": [{"id": 1, "type": "message", "from": "A", "from_id": "u1", "text": "Hi"}]}
        validate_and_load_from_dict(db_session, sample_json)
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

    def test_foreign_key_integrity(self, db_session: Session):
        sample_json = {"id": 404, "messages": [{"id": 30, "type": "message", "from": "T", "from_id": "ut", "text": ""}]}
        validate_and_load_from_dict(db_session, sample_json)
        message = db_session.query(Message).one()
        assert message.chat.telegram_id == 404 and message.author.name == "T"

    def test_cascade_delete(self, db_session: Session):
        sample_json = {"id": 505, "messages": [{"id": 40, "type": "message", "from_id": "ud", "from": "D", "text": ""}]}
        chat = validate_and_load_from_dict(db_session, sample_json)
        db_session.delete(chat)
        db_session.commit()
        assert db_session.query(Message).count() == 0

    def test_unicode_handling(self, db_session: Session):
        unicode_json = {"id": 606, "messages": [{"id": 50, "type": "message", "from": "Мария", "from_id": "um", "text": "Привет! 🌍"}]}
        validate_and_load_from_dict(db_session, unicode_json)
        assert db_session.query(Message).one().text == "Привет! 🌍" and db_session.query(User).one().name == "Мария"

    def test_alias_for_from_field(self, db_session: Session):
        json_with_from = {"id": 303, "messages": [{"id": 20, "type": "message", "from": "Special", "from_id": "us", "text": ""}]}
        validate_and_load_from_dict(db_session, json_with_from)
        assert db_session.query(User).filter_by(telegram_id="us").one().name == "Special"
    
    def test_service_message_handling(self, db_session: Session):
        service_msg_json = {"id": 808, "messages": [{"id": 1, "type": "service"}, {"id": 2, "type": "message", "from_id": "u1", "from": "U"}]}
        validate_and_load_from_dict(db_session, service_msg_json)
        msg1, msg2 = db_session.query(Message).order_by(Message.telegram_id).all()
        assert msg1.is_service and msg1.author_id is None and not msg2.is_service and msg2.author_id is not None

    def test_performance_idempotency_with_cache(self, db_session: Session):
        sample_json = {"id": 707, "messages": [{"id": i, "type": "message", "from_id": f"u{i % 2}", "from": "User"} for i in range(10)]}
        validate_and_load_from_dict(db_session, sample_json)
        assert db_session.query(Message).count() == 10 and db_session.query(User).count() == 2
        
        db_session.add_all = MagicMock()
        validate_and_load_from_dict(db_session, sample_json)
        db_session.add_all.assert_not_called()
        assert db_session.query(Message).count() == 10

    def test_raw_text_storage(self, db_session: Session):
        structured_text = ["Link test: ", {"type": "link", "text": "https://example.com"}]
        json_data = {"id": 1010, "messages": [{"id": 1, "type": "message", "from_id": "u1", "from": "U", "text": structured_text}]}
        validate_and_load_from_dict(db_session, json_data)
        
        msg = db_session.query(Message).one()
        assert msg.text == "Link test: https://example.com"
        assert isinstance(msg.raw_text, list) and msg.raw_text[1]['type'] == 'link'

    # --- Группа 2: Полнота данных и граничные случаи ETL ---

    def test_full_message_data_load(self, db_session: Session):
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
        validate_and_load_from_dict(db_session, full_message_json)
        
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

    def test_batch_loading_logic(self, db_session: Session):
        batch_test_json = {
            "id": 111, "messages": [{"id": i, "type": "message", "from_id": "u1", "from": "U1"} for i in range(12)]
        }
        with patch.object(db_session, 'add_all', wraps=db_session.add_all) as mock_add_all:
            validate_and_load_from_dict(db_session, batch_test_json, batch_size=5)
            assert db_session.query(Message).count() == 12 and db_session.query(User).count() == 1
            assert mock_add_all.call_count == 4

    def test_loading_empty_messages_list(self, db_session: Session):
        empty_json = {"id": 222, "name": "Empty Chat", "messages": []}
        chat = validate_and_load_from_dict(db_session, empty_json)
        assert chat is not None and chat.telegram_id == 222
        assert db_session.query(Message).count() == 0 and db_session.query(User).count() == 0

    def test_adding_new_messages_to_existing_chat(self, db_session: Session):
        initial_json = {"id": 333, "messages": [{"id": 1, "type": "message", "text": "old"}]}
        validate_and_load_from_dict(db_session, initial_json)
        updated_json = {"id": 333, "messages": [{"id": 1, "type": "message", "text": "old"}, {"id": 2, "type": "message", "text": "new"}]}
        validate_and_load_from_dict(db_session, updated_json)
        assert db_session.query(Chat).count() == 1 and db_session.query(Message).count() == 2
        assert {"old", "new"} == {msg.text for msg in db_session.query(Message).all()}

    @patch('database.logging.error')
    def test_invalid_json_validation_fails_gracefully(self, mock_logging_error: MagicMock, db_session: Session):
        invalid_json = {"id": 444, "name": "Invalid Chat"}
        result = validate_and_load_from_dict(db_session, invalid_json)
        assert result is None and db_session.query(Chat).count() == 0
        mock_logging_error.assert_called_once()