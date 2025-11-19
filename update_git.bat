@echo off
setlocal

if "%~1"=="" (
    echo Usa: update_git.bat "mensagem de commit"
    exit /b 1
)

git add .
git commit -m "%*"
git push -u origin main

endlocal
