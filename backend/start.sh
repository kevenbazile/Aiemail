#!/bin/bash
pip install --no-cache-dir -r requirements.txt
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.app:app
