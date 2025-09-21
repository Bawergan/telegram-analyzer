# database.py

# --- 0. Импорты и конфигурация ---
import logging
import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Any, Dict, TypeVar, Tuple, Type

from sqlalchemy.types import UserDefinedType

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field, ValidationError

from sqlalchemy import (
    create_engine, String, DateTime, Text, ForeignKey,
    UniqueConstraint, JSON, Integer, Boolean
)
from sqlalchemy.orm import (
    sessionmaker, Session, DeclarativeBase, Mapped,
    mapped_column, relationship
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Управление конфигурацией ---

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    """
    Класс для управления настройками приложения с использованием Pydantic.
    Загружает конфигурацию из .env файла или использует значения по умолчанию.
    """
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'telegram_archive.db'}"
    MEDIA_STORAGE_ROOT: str = str(BASE_DIR / "media_storage")
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')

settings = Settings()

logging.info(f"Database will be stored at: {settings.DATABASE_URL}")
logging.info(f"Media files will be stored in: {settings.MEDIA_STORAGE_ROOT}")

# --- 2. Схемы Pydantic ---

class ReactionSchema(BaseModel):
    emoji: Optional[str] = None
    count: Optional[int] = None

class TextEntitySchema(BaseModel):
    type: str
    text: str

class LocationSchema(BaseModel):
    latitude: float
    longitude: float
    
class PollSchema(BaseModel):
    question: str
    total_voters: int
    answers: List[Dict[str, Any]]

class ContactInformationSchema(BaseModel):
    phone_number: str
    first_name: str
    last_name: Optional[str] = None
    user_id: Optional[str] = None


class MessageSchema(BaseModel):
    id: int
    type: str
    date: Optional[datetime] = None
    edited: Optional[datetime] = None
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
    location_information: Optional[LocationSchema] = None
    poll: Optional[PollSchema] = None
    contact_information: Optional[ContactInformationSchema] = None
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
    edited_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
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
    stored_media_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    derived_text_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    chat: Mapped["Chat"] = relationship(back_populates="messages")
    author: Mapped[Optional["User"]] = relationship(back_populates="messages")
    text_entities: Mapped[List["TextEntity"]] = relationship(back_populates="message", cascade="all, delete-orphan", passive_deletes=True)
    reactions: Mapped[List["Reaction"]] = relationship(back_populates="message", cascade="all, delete-orphan", passive_deletes=True)
    __table_args__ = (UniqueConstraint('telegram_id', 'chat_id', name='_telegram_id_chat_id_uc'),)

class TextEntity(Base):
    __tablename__ = "text_entities"
    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(String)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    message: Mapped["Message"] = relationship(back_populates="text_entities")

class Reaction(Base):
    __tablename__ = "reactions"
    id: Mapped[int] = mapped_column(primary_key=True)
    emoji: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    message: Mapped["Message"] = relationship(back_populates="reactions")


# --- 5. Интерфейс базы данных и логика ETL ---
def get_or_create(session: Session, model: Type[T], defaults: Optional[dict] = None, **kwargs) -> Tuple[T, bool]:
    """
    Находит существующий экземпляр модели или создает новый.

    Args:
        session: Сессия SQLAlchemy.
        model: Класс ORM-модели.
        defaults: Словарь со значениями для создания нового экземпляра.
        **kwargs: Аргументы для поиска существующего экземпляра.

    Returns:
        Кортеж (instance, created), где 'created' - bool флаг, был ли экземпляр создан.
    """
    instance = session.query(model).filter_by(**kwargs).one_or_none()
    if instance:
        return instance, False
    defaults = defaults or {}
    instance = model(**kwargs, **defaults)
    session.add(instance)
    return instance, True

def _extract_text_and_entities(msg: MessageSchema) -> Tuple[str, List[TextEntitySchema]]:
    """
    Извлекает текстовое содержимое и text_entities из сложной структуры поля 'text'.

    Args:
        msg: Pydantic-схема сообщения (MessageSchema).

    Returns:
        Кортеж (text_content, text_entities_list).
    """
    if not msg.text_entities and isinstance(msg.text, list):
        try:
            entities_from_text = [item for item in msg.text if isinstance(item, dict)]
            text_entities = [TextEntitySchema.model_validate(e) for e in entities_from_text]
        except ValidationError:
            logging.warning(f"Could not parse text_entities from structured text for message {msg.id}")
            text_entities = []
    else:
        text_entities = msg.text_entities or []
    if isinstance(msg.text, list):
        text_content = "".join([item.get('text', str(item)) if isinstance(item, dict) else str(item) for item in msg.text])
    else:
        text_content = str(msg.text) if msg.text else ""
    return text_content, text_entities

def _handle_media_file(
    message_orm: Message,
    msg_schema: MessageSchema,
    chat_telegram_id: int,
    export_root: str,
    media_storage_root: str
):
    """
    Обрабатывает медиафайл, связанный с сообщением: копирует его в центральное
    хранилище и сохраняет новый путь в ORM-объекте.

    Args:
        message_orm: ORM-объект сообщения для обновления.
        msg_schema: Pydantic-схема сообщения, содержащая информацию о файле.
        chat_telegram_id: Telegram ID чата для формирования уникального имени файла.
        export_root: Корневая директория конкретного экспорта.
        media_storage_root: Корневая директория центрального медиахранилища.
    """
    if not msg_schema.file:
        return

    source_path = Path(export_root) / msg_schema.file
    if source_path.exists():
        extension = source_path.suffix
        new_filename = f"{chat_telegram_id}_{msg_schema.id}{extension}"
        dest_path = Path(media_storage_root) / new_filename
        
        try:
            shutil.copy2(source_path, dest_path)
            message_orm.stored_media_path = new_filename
        except Exception:
            logging.error(f"Failed to copy file {source_path} to {dest_path}", exc_info=True)
    else:
        logging.warning(f"Media file not found for message {msg_schema.id}: {source_path}")


def _populate_message_from_schema(
    message_orm: Message,
    msg_schema: MessageSchema,
    chat_telegram_id: int,
    export_root: str,
    media_storage_root: str
):
    """
    Наполняет ORM-объект Message данными из Pydantic-схемы, включая обработку медиафайлов.

    Args:
        message_orm: ORM-объект сообщения (новый или существующий).
        msg_schema: Pydantic-схема сообщения с исходными данными.
        chat_telegram_id: Telegram ID чата.
        export_root: Путь к директории с исходным экспортом.
        media_storage_root: Путь к центральному хранилищу медиафайлов.
    """
    text_content, text_entities_schemas = _extract_text_and_entities(msg_schema)
    
    message_orm.text = text_content
    message_orm.raw_text = msg_schema.model_dump().get('text')
    message_orm.edited_date = msg_schema.edited
    message_orm.date = msg_schema.date
    message_orm.reply_to_message_id = msg_schema.reply_to_message_id
    message_orm.media_type = msg_schema.media_type
    message_orm.mime_type = msg_schema.mime_type
    message_orm.duration_seconds = msg_schema.duration_seconds
    message_orm.file_path = msg_schema.file
    
    message_orm.text_entities.clear()
    message_orm.reactions.clear()
    message_orm.text_entities = [TextEntity(type=e.type, text=e.text) for e in text_entities_schemas]
    message_orm.reactions = [Reaction(emoji=r.emoji, count=r.count) for r in msg_schema.reactions or []]

    _handle_media_file(message_orm, msg_schema, chat_telegram_id, export_root, media_storage_root)


def load_validated_data(
    db: Session,
    chat_data: TelegramChatExportSchema,
    export_root: str,
    media_storage_root: str,
    batch_size: int = 1000
) -> Tuple[Optional[Chat], int]:
    """
    Загружает валидированные данные из Pydantic-схемы в базу данных.
    Реализует логику upsert для чатов, пользователей и сообщений.

    Args:
        db: Сессия SQLAlchemy.
        chat_data: Pydantic-схема с данными чата.
        export_root: Путь к директории с исходным экспортом.
        media_storage_root: Путь к центральному хранилищу медиафайлов.
        batch_size: Размер пакета для массовой вставки сообщений.

    Returns:
        Кортеж (chat_object, new_messages_count), где new_messages_count -
        количество реально добавленных в БД новых сообщений.
    """
    chat, created = get_or_create(db, Chat, telegram_id=chat_data.id, defaults={'name': chat_data.name, 'type': chat_data.type})
    if not created and (chat.name != chat_data.name or chat.type != chat_data.type):
        logging.info(f"Updating metadata for chat '{chat.name}' -> '{chat_data.name}' (ID: {chat.telegram_id})")
        chat.name = chat_data.name
        chat.type = chat_data.type
    
    db.flush()
    
    existing_messages_map = {m.telegram_id: m for m in db.query(Message).filter_by(chat_id=chat.id)}
    user_cache = {user.telegram_id: user for user in db.query(User)}
    logging.info(f"Found {len(existing_messages_map)} existing messages and {len(user_cache)} users in DB for this context.")
    
    messages_to_process = []
    users_to_create = {}

    for msg_schema in chat_data.messages:
        existing_message = existing_messages_map.get(msg_schema.id)
        
        is_edited = msg_schema.edited and (not existing_message or not existing_message.edited_date or msg_schema.edited > existing_message.edited_date)
        if existing_message and not is_edited:
            continue
        
        if existing_message and is_edited:
            logging.info(f"Updating message {msg_schema.id} due to new edit date: {msg_schema.edited}")
            _populate_message_from_schema(existing_message, msg_schema, chat_data.id, export_root, media_storage_root)
            continue
        
        author_telegram_id = msg_schema.from_id
        if author_telegram_id:
            user_in_db = user_cache.get(author_telegram_id) or users_to_create.get(author_telegram_id)
            if user_in_db and user_in_db.name != msg_schema.from_name:
                logging.info(f"Updating user name '{user_in_db.name}' -> '{msg_schema.from_name}' (ID: {author_telegram_id})")
                user_in_db.name = msg_schema.from_name
            elif not user_in_db:
                users_to_create[author_telegram_id] = User(telegram_id=author_telegram_id, name=msg_schema.from_name)

        new_message = Message(telegram_id=msg_schema.id, is_service=(msg_schema.type == 'service'), chat_id=chat.id)
        _populate_message_from_schema(new_message, msg_schema, chat_data.id, export_root, media_storage_root)
        messages_to_process.append((new_message, author_telegram_id))

    if users_to_create:
        logging.info(f"Creating {len(users_to_create)} new users.")
        db.add_all(users_to_create.values())
        db.flush()
        for user in users_to_create.values():
            user_cache[user.telegram_id] = user

    new_messages_count = 0
    messages_to_add = []
    for message, author_telegram_id in messages_to_process:
        if author_telegram_id and author_telegram_id in user_cache:
            message.author_id = user_cache[author_telegram_id].id
        
        messages_to_add.append(message)
        if len(messages_to_add) >= batch_size:
            logging.info(f"Adding batch of {len(messages_to_add)} messages to session.")
            db.add_all(messages_to_add)
            new_messages_count += len(messages_to_add)
            messages_to_add = []
            
    if messages_to_add:
        logging.info(f"Adding final batch of {len(messages_to_add)} messages to session.")
        db.add_all(messages_to_add)
        new_messages_count += len(messages_to_add)

    logging.info(f"Committing changes to the database. Total new messages: {new_messages_count}.")
    db.commit()
    return chat, new_messages_count

def validate_and_load_from_dict(
    db: Session,
    raw_data: dict,
    export_root: str,
    media_storage_root: str,
    batch_size: int = 1000
) -> Tuple[Optional[Chat], int]:
    """
    Валидирует 'сырые' данные из JSON и запускает процесс их загрузки в БД.

    Args:
        db: Сессия SQLAlchemy.
        raw_data: Словарь с данными из result.json.
        export_root: Путь к директории с исходным экспортом.
        media_storage_root: Путь к центральному хранилищу медиафайлов.
        batch_size: Размер пакета для массовой вставки.

    Returns:
        Кортеж (chat_object, new_messages_count) или (None, 0) в случае ошибки валидации.
    """
    try:
        validated_chat_export = TelegramChatExportSchema.model_validate(raw_data)
        return load_validated_data(db, validated_chat_export, export_root, media_storage_root, batch_size)
    except ValidationError:
        logging.error("JSON validation failed.", exc_info=True)
        return None, 0

# --- 7. Точка входа для запуска скрипта ---
def main(json_path: str, export_root: str):
    """
    Основная функция для запуска ETL-процесса для одного файла result.json.
    """
    logging.info(f"Starting ETL process for file: {json_path}")
    
    if not os.path.exists(json_path):
        logging.error(f"File not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    db: Session = SessionLocal()
    try:
        Base.metadata.create_all(bind=engine)
        
        chat, new_count = validate_and_load_from_dict(db, raw_data, export_root, settings.MEDIA_STORAGE_ROOT)
        if chat:
            logging.info(f"Successfully processed data for chat: '{chat.name}' (ID: {chat.telegram_id}). Added {new_count} new messages.")
        else:
            logging.error("ETL process failed.")
    finally:
        db.close()
        logging.info("Database session closed.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_to_process = Path(sys.argv[1]).resolve()
        export_root_dir = file_to_process.parent
        main(str(file_to_process), str(export_root_dir))
    else:
        print("Usage: python database.py <path_to_result.json>")