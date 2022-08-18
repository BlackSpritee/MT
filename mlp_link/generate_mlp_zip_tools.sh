#!/usr/bin/env bash

zip_name="udf_mlp.zip"
zip_name2="udf_user_code.zip"

# tar tools
if [ -f $zip_name ];then
    rm -f $zip_name
fi
if [ -f $zip_name2 ];then
    rm -f $zip_name2
fi
zip -r $zip_name ./udf_mlp/*

#用户代码，从根目录开始打包，剔除隐藏文件，用户可通过 -x剔除指定目录
zip -r $zip_name2 ./  -x "./udf_mlp/*" -x "*/\.*" -x "\.*" -x "__pycache__"
