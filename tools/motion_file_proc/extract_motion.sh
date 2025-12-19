#!/bin/bash

# 检查参数数量
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_directory> [output_directory]"
    echo "Example: $0 Data/motion_data/my_dataset [Data/cleaned_dataset]"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# 获取当前脚本所在的目录 (motion_scripts)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 定位 Python 脚本的路径 (相对于 motion_scripts 目录)
PYTHON_TOOL="$SCRIPT_DIR/../PARC/tools/pkl_tools/pkl_extract_frames_fps.py"

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_TOOL" ]; then
    echo "Error: Could not find python script at: $PYTHON_TOOL"
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

echo "Starting extraction..."
echo "Input Directory: $INPUT_DIR"
if [ ! -z "$OUTPUT_DIR" ]; then
    echo "Output Directory: $OUTPUT_DIR"
else
    echo "Output: In-place (suffix: _cleaned)"
fi
echo "Using Tool: $PYTHON_TOOL"
echo "----------------------------------------"

# 构建命令
CMD="python \"$PYTHON_TOOL\" --input \"$INPUT_DIR\""

if [ ! -z "$OUTPUT_DIR" ]; then
    CMD="$CMD --output \"$OUTPUT_DIR\""
fi

# 执行命令
eval $CMD

echo "----------------------------------------"
echo "Done."
