"""Replay actual downloaded task resources inside a network-isolated sandbox."""
from email.message import Message
import hashlib
from io import BytesIO
import json
from pathlib import Path
import urllib.error
import urllib.request
from urllib.response import addinfourl


def install():
    folder = Path(__file__).parent / "assets"
    manifest = json.loads((folder / "manifest.json").read_text())
    resources = {}
    for name, row in manifest["resources"].items():
        data = (folder / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise RuntimeError(f"Offline task resource changed: {name}")
        for url in row["aliases"]:
            resources[url] = data

    def urlopen(url, *args, **kwargs):
        address = url.full_url if isinstance(url, urllib.request.Request) else str(url)
        if address not in resources:
            raise urllib.error.URLError("External network disabled; URL has no frozen task resource")
        headers = Message()
        headers["Content-Length"] = str(len(resources[address]))
        return addinfourl(BytesIO(resources[address]), headers, address, 200)

    import requests

    def request(session, method, url, *args, **kwargs):
        if method.upper() != "GET" or url not in resources:
            raise requests.exceptions.ConnectionError("External network disabled; URL has no frozen task resource")
        response = requests.Response()
        response.status_code, response.url = 200, url
        response._content = resources[url]
        response._content_consumed = True
        response.headers["Content-Length"] = str(len(response._content))
        return response

    urllib.request.urlopen = urlopen
    requests.sessions.Session.request = request
    return {name: row["sha256"] for name, row in manifest["resources"].items()}
