@echo off
setlocal

echo Creating virtual environment...
python -m venv venv || goto :error

echo.
echo Activating environment and upgrading pip...
call venv\Scripts\activate.bat || goto :error
pip install --upgrade pip || goto :error

echo.
echo Installing requirements...
pip install -r requirements.txt || goto :error

echo.
echo =================================================================
echo Setup completed successfully!
echo To activate again later: venv\Scripts\activate
echo To run: python h1-dyadic-law.py --n_dyads 500 --seed 42
echo =================================================================
goto :end

:error
echo.
echo =================================================================
echo ERROR: Setup failed! See the messages above for details.
echo =================================================================
pause
exit /b 1

:end
pause
