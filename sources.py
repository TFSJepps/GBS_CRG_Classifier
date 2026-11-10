import base64
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
GRAPH = "https://graph.microsoft.com/v1.0"
SCOPES = ["Files.Read.All", "Sites.Read.All"]
THUMB_SIZE = "c384x384"  # scaled to fit (no crop); SigLIP resizes to 384 px anyway
RETRY_STATUS = {429, 500, 502, 503, 504}


class LocalImage:
    def __init__(self, path, root):
        self.path = path
        self.rel = str(path.relative_to(root))
        self.ref = str(path)

    def load(self):
        return Image.open(self.path)


class LocalSource:
    def __init__(self, folder):
        self.folder = folder

    def list_images(self):
        paths = sorted(p for p in self.folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())
        return [LocalImage(p, self.folder) for p in paths]


class GraphImage:
    def __init__(self, source, item_id, rel, web_url):
        self.source = source
        self.item_id = item_id
        self.rel = rel
        self.ref = web_url

    def load(self):
        return Image.open(io.BytesIO(self.source.fetch(self.item_id)))


class GraphSource:
    """Streams images from a SharePoint/OneDrive folder via Microsoft Graph; nothing is written to disk."""

    def __init__(self, share_url, cache_path, thumbnails=False):
        try:
            import truststore
            truststore.inject_into_ssl()  # trust the Windows cert store, so corporate TLS-inspection proxies verify
            import msal
            import requests
        except ImportError as e:
            sys.exit(f"{e.name} is not installed in this Python; run: {sys.executable} -m pip install msal requests truststore")
        self.requests = requests
        config_path = Path(__file__).resolve().parent / "graph_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        client_id = os.environ.get("GRAPH_CLIENT_ID") or config.get("client_id")
        if not client_id:
            sys.exit(f"Put your Azure app's client_id and tenant_id in {config_path} (or set GRAPH_CLIENT_ID / GRAPH_TENANT_ID).")
        tenant = os.environ.get("GRAPH_TENANT_ID") or config.get("tenant_id") or "organizations"

        self.share_url = share_url
        self.thumbnails = thumbnails
        self.cache_path = cache_path
        self.cache = msal.SerializableTokenCache()
        if cache_path.exists():
            self.cache.deserialize(cache_path.read_text(encoding="utf-8"))
        self.app = msal.PublicClientApplication(
            client_id, authority=f"https://login.microsoftonline.com/{tenant}", token_cache=self.cache
        )
        self.lock = threading.Lock()
        self.local = threading.local()
        self.drive_id = None

    def _token(self, interactive=False):
        with self.lock:
            result = None
            accounts = self.app.get_accounts()
            if accounts:
                result = self.app.acquire_token_silent(SCOPES, account=accounts[0])
            if not result and interactive:
                result = self.app.acquire_token_interactive(SCOPES)
            if not result or "access_token" not in result:
                detail = result.get("error_description") if result else "no cached sign-in"
                raise RuntimeError(f"Microsoft sign-in failed: {detail}")
            if self.cache.has_state_changed:
                self.cache_path.write_text(self.cache.serialize(), encoding="utf-8")
            return result["access_token"]

    def _session(self):
        if not hasattr(self.local, "session"):
            self.local.session = self.requests.Session()
        return self.local.session

    def _request(self, url, allow_404=False):
        for attempt in range(6):
            wait = min(2**attempt, 60)
            try:
                r = self._session().get(url, headers={"Authorization": f"Bearer {self._token()}"}, timeout=60)
            except self.requests.RequestException:
                time.sleep(wait)
                continue
            if r.status_code == 404 and allow_404:
                return None
            if r.status_code in RETRY_STATUS or r.status_code == 401:
                time.sleep(float(r.headers.get("Retry-After", wait)))
                continue
            r.raise_for_status()
            return r
        raise RuntimeError(f"Giving up after repeated failures: {url}")

    def fetch(self, item_id):
        base = f"{GRAPH}/drives/{self.drive_id}/items/{item_id}"
        if self.thumbnails:
            r = self._request(f"{base}/thumbnails/0/{THUMB_SIZE}/content", allow_404=True)
            if r is not None:
                return r.content
        return self._request(f"{base}/content").content

    def list_images(self):
        self._token(interactive=True)
        encoded = "u!" + base64.urlsafe_b64encode(self.share_url.encode()).decode().rstrip("=")
        root = self._request(f"{GRAPH}/shares/{encoded}/driveItem").json()
        if "folder" not in root:
            sys.exit("The link must point to a folder, not a single file.")
        self.drive_id = root["parentReference"]["driveId"]

        images = []
        stack = [(root["id"], "")]
        while stack:
            folder_id, prefix = stack.pop()
            url = f"{GRAPH}/drives/{self.drive_id}/items/{folder_id}/children?$top=999"
            while url:
                page = self._request(url).json()
                for item in page["value"]:
                    rel = f"{prefix}{item['name']}"
                    if "folder" in item:
                        stack.append((item["id"], rel + "/"))
                    elif Path(item["name"]).suffix.lower() in IMAGE_EXTS:
                        images.append(GraphImage(self, item["id"], rel, item.get("webUrl", "")))
                url = page.get("@odata.nextLink")
            print(f"\rListing... {len(images)} images found", end="", flush=True)
        print()
        return sorted(images, key=lambda im: im.rel)
