"""Загрузка файла в S3 и выдача временной ссылки для воркера."""
import os
import uuid
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.client import Config

import config

_s3 = boto3.client(
    "s3", endpoint_url=config.S3_ENDPOINT,
    aws_access_key_id=config.S3_ACCESS_KEY, aws_secret_access_key=config.S3_SECRET_KEY,
    region_name=config.S3_REGION, config=Config(signature_version="s3v4"),
)

# Многочастная заливка: файл читается кусками по 8 МБ, целиком в памяти не держится.
# Без этого двухчасовое видео на 1,5 ГБ клало бы бота по RAM.
_TRANSFER = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,
    multipart_chunksize=8 * 1024 * 1024,
    max_concurrency=4,
    use_threads=True,
)


def _presign(key: str) -> str:
    return _s3.generate_presigned_url(
        "get_object", Params={"Bucket": config.S3_BUCKET, "Key": key},
        ExpiresIn=config.PRESIGN_TTL,
    )


def upload_bytes(data: bytes, filename: str) -> str:
    """Заливка из памяти. Осталась для мелочи; большие файлы — через upload_file."""
    key = f"tg/{uuid.uuid4().hex}/{filename}"
    _s3.put_object(Bucket=config.S3_BUCKET, Key=key, Body=data)
    return _presign(key)


def upload_file(path: str | os.PathLike, filename: str | None = None) -> str:
    """Заливка файла с диска потоком. Блокирующая — из async-кода вызывать через
    asyncio.to_thread, иначе на время заливки встанет весь бот."""
    name = filename or os.path.basename(os.fspath(path))
    key = f"tg/{uuid.uuid4().hex}/{name}"
    _s3.upload_file(os.fspath(path), config.S3_BUCKET, key, Config=_TRANSFER)
    return _presign(key)