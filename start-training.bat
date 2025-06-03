@echo off
cls
echo ============================================================
echo                  YANIV AI TRAINING SYSTEM
echo ============================================================
echo.
echo Choose training duration:
echo.
echo   1. Quick Test   (20 episodes  - 2 minutes)
echo   2. Short        (100 episodes - 10 minutes)  
echo   3. Medium       (500 episodes - 45 minutes)
echo   4. Full         (4000 episodes - 3-4 hours)
echo.
set /p choice="Enter your choice (1-4): "

if %choice%==1 set episodes=20
if %choice%==2 set episodes=100
if %choice%==3 set episodes=500
if %choice%==4 set episodes=4000

echo.
echo ============================================================
echo Starting training for %episodes% episodes...
echo.
echo IMPORTANT: Progress will appear after first 10 episodes
echo The training IS running even if you don't see updates!
echo.
echo Press Ctrl+C to stop at any time.
echo ============================================================
echo.

node train-progress-bar.js %episodes%