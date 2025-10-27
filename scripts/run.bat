@echo off
REM Windows batch script to run Transformer training

REM Set Python path
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Run training
python -m src.train

REM Optional: Use config file
REM python -m src.train --config configs/base_config.yaml --seed 42

echo Training completed!

