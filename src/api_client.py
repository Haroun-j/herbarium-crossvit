"""Client for the image server used during the project.

The herbarium images are confidential (course data), so they are not in this
repository. During the project they were served by a small HTTPS API protected
by a client certificate and basic auth. The address and the credentials are
read from environment variables (see .env.example).

Endpoints:
    GET /api/ids            -> {"ids": [...]}
    GET /api/meta/<id>      -> {"id", "classic_file", "seg_file", "meta": {"nom", "epines"}}
    GET /api/classic/<id>   -> full herbarium sheet (jpeg or png)
    GET /api/seg/<id>       -> segmented image, plant on a black background
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageFile
from requests.adapters import HTTPAdapter

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_dotenv(path=".env"):
    """Small .env reader, so python-dotenv is not needed."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _ext_from_content_type(content_type):
    ct = content_type.split(";")[0].strip().lower()
    if ct == "image/jpeg":
        return ".jpg"
    if ct == "image/png":
        return ".png"
    return ".img"


class HerbariumClient:
    def __init__(self, base_url=None, cache_dir="img_cache", timeout=30, max_retries=3):
        load_dotenv()
        self.base_url = (base_url or os.environ.get("HERBARIUM_API_URL", "")).rstrip("/")
        if not self.base_url:
            raise RuntimeError("HERBARIUM_API_URL is not set, see .env.example")

        cert = os.environ.get("HERBARIUM_CLIENT_CERT")
        key = os.environ.get("HERBARIUM_CLIENT_KEY")
        self.cert = (cert, key) if cert and key else None
        self.verify = os.environ.get("HERBARIUM_CA_CERT") or True
        user = os.environ.get("HERBARIUM_API_USER")
        self.auth = (user, os.environ.get("HERBARIUM_API_PASSWORD", "")) if user else None

        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=0)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get(self, path, **kwargs):
        return self.session.get(self.base_url + path, cert=self.cert, verify=self.verify,
                                auth=self.auth, timeout=self.timeout, **kwargs)

    def list_ids(self):
        r = self._get("/api/ids")
        r.raise_for_status()
        return r.json()["ids"]

    def get_meta(self, img_id):
        r = self._get(f"/api/meta/{img_id}")
        r.raise_for_status()
        return r.json()

    def _cached_file(self, img_id, kind):
        for ext in (".jpg", ".jpeg", ".png", ".img"):
            p = self.cache_dir / f"{kind}_{img_id}{ext}"
            if p.exists() and p.stat().st_size > 0:
                return p
        return None

    def fetch_image(self, img_id, kind="classic"):
        """Return a PIL image in RGB. kind is 'classic' (full sheet) or 'seg'."""
        cached = self._cached_file(img_id, kind)
        if cached is not None:
            try:
                img = Image.open(cached)
                img.load()
                return img.convert("RGB")
            except Exception:
                cached.unlink(missing_ok=True)  # broken download, fetch it again

        last_error = None
        for attempt in range(self.max_retries):
            try:
                with self._get(f"/api/{kind}/{img_id}", stream=True) as r:
                    r.raise_for_status()
                    ext = _ext_from_content_type(r.headers.get("Content-Type", ""))
                    target = self.cache_dir / f"{kind}_{img_id}{ext}"
                    tmp = target.with_suffix(target.suffix + ".tmp")
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    if tmp.stat().st_size == 0:
                        raise IOError("empty file")
                    os.replace(tmp, target)
                img = Image.open(target)
                img.load()
                return img.convert("RGB")
            except Exception as e:
                last_error = e
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"could not fetch {kind}/{img_id}: {last_error}")

    def fetch_many(self, img_ids, kind="classic", workers=16):
        """Fetch several images in parallel. Returns {img_id: PIL image}."""
        results = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(self.fetch_image, i, kind): i for i in img_ids}
            for f in as_completed(futures):
                img_id = futures[f]
                try:
                    results[img_id] = f.result()
                except Exception as e:
                    print(f"fetch failed for {kind}/{img_id}: {e}")
        return results
