"""Загрузка файла в S3 и выдача временной ссылки для воркера."""
import uuid
import boto3
from botocore.client import Config

import config

_s3 = boto3.client(
    "s3", endpoint_url=config.S3_ENDPOINT,
    aws_access_key_id=config.S3_ACCESS_KEY, aws_secret_access_key=config.S3_SECRET_KEY,
    region_name=config.S3_REGION, config=Config(signature_version="s3v4"),
)


def upload_bytes(data: bytes, filename: str) -> str:
    key = f"tg/{uuid.uuid4().hex}/{filename}"
    _s3.put_object(Bucket=config.S3_BUCKET, Key=key, Body=data)
    return _s3.generate_presigned_url(
        "get_object", Params={"Bucket": config.S3_BUCKET, "Key": key},
        ExpiresIn=config.PRESIGN_TTL,
    )
