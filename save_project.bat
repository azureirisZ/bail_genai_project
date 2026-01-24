@echo off
echo 🚀 Saving progress to GitHub...
git add .
set /p "msg=Enter commit message (or press Enter for 'Update'): " || set "msg=Update"
git commit -m "%msg%"
git push
echo ✅ Done! Your code is safe on GitHub.
pause