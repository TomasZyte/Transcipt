"""
Хранилище загруженных файлов. S3-совместимое (Cloudflare R2 / Backblaze B2 / MinIO).
Воркер (в т.ч. удалённый на RunPod) скачивает файл по временной presigned-ссылке.
"""

import os
import uuid
import boto3
from botocore.client import Config

S3_ENDPOINT = os.getenv("S3_ENDPOINT")            # напр. https://<acct>.r2.cloudflarestorage.com
S3_BUCKET = os.getenv("S3_BUCKET", "transcribe")
S3_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET = os.getenv("S3_SECRET_KEY")
S3_REGION = os.getenv("S3_REGION", "auto")
PRESIGN_TTL = int(os.getenv("PRESIGN_TTL", "3600"))

_s3 = boto3.client(
    "s3", endpoint_url=S3_ENDPOINT, aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET, region_name=S3_REGION,
    config=Config(signature_version="s3v4"),
)


def upload(data: bytes, filename: str) -> str:
    key = f"uploads/{uuid.uuid4().hex}/{filename}"
    _s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data)
    return key


def presigned_url(key: str) -> str:
    return _s3.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=PRESIGN_TTL
    )
