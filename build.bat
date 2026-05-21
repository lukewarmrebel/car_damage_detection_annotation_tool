@echo off
echo Installing PyInstaller...
pip install pyinstaller

echo.
echo Building executable...
pyinstaller ^
  --onefile ^
  --windowed ^
  --name "CarDamageAnnotationTool" ^
  --add-data "best.pt;." ^
  --add-data "class_dict.txt;." ^
  Car_damage_annotation_compression_tool_with_AI_damage_detection.py

echo.
echo Done. Executable is in the dist\ folder.
pause
