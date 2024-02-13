@echo off
echo Setting up virtual environment...
python -m venv myenv

echo Activating virtual environment...
call myenv\Scripts\activate

echo Installing required packages...
pip install -r requirements.txt

echo Setup complete.
pause
