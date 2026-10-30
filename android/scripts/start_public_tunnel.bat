@echo off
title Photovance AI - Public Cloudflare HTTPS Tunnel (Port 7870)
echo ======================================================================
echo    PHOTOVANCE AI - PUBLIC CLOUDFLARE HTTPS TUNNEL (PORT 7870)
echo ======================================================================
echo.
echo Connecting Face Studio (port 7870) to a secure public HTTPS endpoint...
echo Once connected, copy the "https://....trycloudflare.com" URL and enter it
echo into the Photovance Android App Settings (or paste in Google Play Console).
echo.
if exist "C:\Program Files (x86)\cloudflared\cloudflared.exe" (
    "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:7870
) else (
    cloudflared.exe tunnel --url http://127.0.0.1:7870
)
pause
