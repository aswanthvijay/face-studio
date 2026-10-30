@echo off
title Face Studio & Photovance AI - Public Cloudflare HTTPS Tunnel
echo ======================================================================
echo    FACE STUDIO - PUBLIC CLOUDFLARE HTTPS TUNNEL (PORT 7870)
echo ======================================================================
echo.
echo Creating zero-config worldwide HTTPS tunnel for Port 7870...
if exist "C:\Program Files (x86)\cloudflared\cloudflared.exe" (
    "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:7870
) else (
    cloudflared.exe tunnel --url http://127.0.0.1:7870
)
pause
