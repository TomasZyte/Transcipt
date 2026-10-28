"""Explicit, fail-closed configuration for integration_test.py; no API calls."""
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
from urllib.parse import urlsplit

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_BOT_ID = 8831823389
PRODUCTION_ENDPOINT = "l20o867nueti1d"
MAX_BYTES = 10 * 1024 * 1024
MAX_DURATION = 60
EXECUTION_MS = 600_000
TTL_MS = 900_000
SESSION_SECONDS = 1200
RESERVE_USD = Decimal("0.50")  # Planning allowance, not a provider billing cap.


@dataclass(frozen=True)
class TestSettings:
    bot_id: int
    username: str
    user_id: int
    endpoint: str
    bucket: str
    storage_host: str
    runtime: Path
    max_jobs: int
    budget: Decimal
    prior_spend: Decimal
    secrets: dict = field(repr=False)

    @classmethod
    def load(cls, path: Path):
        path = path.resolve(strict=True)
        values = dotenv_values(path, interpolate=False)

        def required(key):
            value = values.get(key, "") or ""
            if not value.strip() or "<" in value or "${" in value:
                raise ValueError(f"Set {key} explicitly in the test env file")
            return value.strip()

        bot_id = int(required("TEST_BOT_ID"))
        user_id = int(required("TEST_ALLOWED_USER_ID"))
        token = required("BOT_TOKEN")
        if (bot_id <= 0 or user_id <= 0 or bot_id == PRODUCTION_BOT_ID
                or not re.fullmatch(r"\d+:[A-Za-z0-9_-]{30,}", token)
                or token.split(":")[0] != str(bot_id)):
            raise ValueError("Unexpected test bot token or user ID")
        username = required("TEST_BOT_USERNAME").lstrip("@").lower()
        if not re.fullmatch(r"[a-z0-9_]+bot", username) or username == "poslovnicabot":
            raise ValueError("Use a separate test bot username")
        endpoint = required("RUNPOD_ENDPOINT_ID")
        if not re.fullmatch(r"[a-z0-9]{10,32}", endpoint) or endpoint == PRODUCTION_ENDPOINT:
            raise ValueError("Use a separate test RunPod endpoint")
        bucket = required("S3_BUCKET")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,54}-test", bucket):
            raise ValueError("The dedicated R2 bucket name must end with -test")
        storage = urlsplit(required("S3_ENDPOINT"))
        if (storage.scheme != "https" or not re.fullmatch(
                r"[a-f0-9]{32}\.r2\.cloudflarestorage\.com", storage.netloc)
                or storage.path not in ("", "/") or storage.query or storage.fragment):
            raise ValueError("S3_ENDPOINT must be an HTTPS R2 account endpoint")
        runtime = ROOT.resolve() / ".test-runtime" / str(bot_id)
        if runtime.resolve() != runtime or (runtime / "bot.db").resolve() != runtime / "bot.db":
            raise ValueError("Test runtime and database must not redirect through symlinks")
        if values.get("DB_PATH") and Path(values["DB_PATH"]).resolve() != runtime / "bot.db":
            raise ValueError("Remove DB_PATH: the test runner owns its isolated database path")
        for key in ("PROVIDER_TOKEN", "LOCAL_BOT_API"):
            if values.get(key):
                raise ValueError(f"{key} must be empty in test mode")
        if values.get("STARS_ENABLED", "0") != "0":
            raise ValueError("STARS_ENABLED must be 0 in test mode")
        try:
            budget = Decimal(required("TEST_BUDGET_USD"))
            prior = Decimal(required("TEST_PRIOR_SPEND_USD"))
            jobs = int(required("TEST_MAX_JOBS"))
            valid = (budget.is_finite() and prior.is_finite() and Decimal("0") <= prior < budget <= 2
                     and 1 <= jobs <= 2 and prior + RESERVE_USD <= budget)
        except (ValueError, InvalidOperation):
            valid = False
        if not valid:
            raise ValueError("Invalid test budget (up to $2), prior spend, or job limit (1-2)")
        secrets = {key: required(key) for key in
                   ("BOT_TOKEN", "RUNPOD_API_KEY", "S3_ACCESS_KEY", "S3_SECRET_KEY")}
        return cls(bot_id, username, user_id, endpoint, bucket, storage.netloc,
                   runtime, jobs, budget, prior, secrets)

    def environment(self):
        # An allowlist: never inherit production app settings or unknown env-file keys.
        return {**self.secrets, "RUNPOD_ENDPOINT_ID": self.endpoint,
                "S3_ENDPOINT": "https://" + self.storage_host, "S3_BUCKET": self.bucket,
                "S3_REGION": "auto", "PRESIGN_TTL": "3600",
                "DB_PATH": str(self.runtime / "bot.db"), "ADMINS": str(self.user_id),
                "LOCAL_BOT_API": "", "PROVIDER_TOKEN": "", "STARS_ENABLED": "0",
                "TG_TIMEOUT": "60", "SOCIAL_PROOF_MIN": "0"}
