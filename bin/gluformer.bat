@echo off
REM 切换到脚本所在目录（保证相对路径能正确执行）
cd /d %~dp0

REM 运行 dubosson 数据集
python ..\lib\gluformer.py --dataset dubosson --gpu_id 0 --optuna True > .\output\track_gluformer_dubosson.txt

REM 运行 hall 数据集
python .\lib\gluformer.py --dataset hall --gpu_id 0 --optuna True > .\output\track_gluformer_hall.txt

pause
