@echo off
title Seleziona modalita di avvio

:menu
cls
echo Scegli come avviare l'applicazione:
echo [1] Avvia in locale (python src\run_all.py)
echo [2] Avvia in cloud (Streamlit)
echo.

set /p scelta="Digita 1 per Locale oppure 2 per Cloud e premi Invio: "

if "%scelta%"=="1" goto locale
if "%scelta%"=="2" goto cloud
goto menu

:locale
echo.
echo Avvio in locale...
python src\run_all.py
pause
goto fine

:cloud
echo.
echo Apertura applicazione nel browser...
start "" "https://romano-luigi-projectwork-28.streamlit.app/"
goto fine

:fine