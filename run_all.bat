@echo off
setlocal
py -m pip install -r requirements.txt || exit /b 1
py -m pytest -q || exit /b 1
py run_all.py --from-raw || exit /b 1
echo.
echo SIG full reproduction completed successfully.
endlocal
