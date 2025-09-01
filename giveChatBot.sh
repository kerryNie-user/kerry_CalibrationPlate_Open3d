#!/bin/bash

output="GiveChatBot.txt"
src="src"

# 清空/创建输出文件
> "$output"

# 只查找文本文件（这里举例拼接 .py 和 .txt，可以自己加扩展名）
find "$src" -type f \( -name "*.py" -o -name "*.txt" -o -name "*.cpp" -o -name "*.hpp" \) | while read -r file; do
    echo "===== $file =====" >> "$output"
    cat "$file" >> "$output"
    echo -e "\n" >> "$output"
done

echo "已生成 $output"