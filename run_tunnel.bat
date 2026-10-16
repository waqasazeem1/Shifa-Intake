@echo off
echo ===================================================
echo Starting Cloudflare Public Demo Tunnel for Shifa Intake
echo Forwarding to local app at http://127.0.0.1:8000
echo ===================================================
cloudflared.exe tunnel --url http://127.0.0.1:8000
pause
