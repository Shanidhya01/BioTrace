from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
resp = client.get('/')
print('status_code=', resp.status_code)
print('json=', resp.json())
