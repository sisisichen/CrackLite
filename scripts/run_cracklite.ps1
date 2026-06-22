param(
    [string]$DataRoot = "data"
)

$ErrorActionPreference = "Stop"
$env:CRACKLITE_DATA_DIR = $DataRoot

python train.py
python evaluate.py
python predict.py
