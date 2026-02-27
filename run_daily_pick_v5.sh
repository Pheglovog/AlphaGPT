#!/bin/bash
# AlphaGPT v5 每日选股 + 持仓分析

cd /Users/hsh/AIGC/AlphaGPT

export PATH="/Users/hsh/Library/Python/3.9/bin:$PATH"
export TUSHARE_TOKEN="cc9f4227a4be5c67699791c24526d2ec3947877f1cec3619866078f4"

# 运行 v5
/usr/bin/python3 /Users/hsh/AIGC/AlphaGPT/daily_pick_v5.py >> /Users/hsh/AIGC/AlphaGPT/daily_pick.log 2>&1

# 读取结果发送
RESULT=$(cat /Users/hsh/AIGC/AlphaGPT/daily_pick_result_v5.txt 2>/dev/null | head -80)

curl -s -X POST "http://127.0.0.1:18789/api/message" \
  -H "Authorization: Bearer 06abb9963441f469fe6c5343accfc51e19486797afec1e51" \
  -H "Content-Type: application/json" \
  -d "{\"channel\": \"feishu\", \"target\": \"user:ou_c53ff42237108108087d63bdc539cf96\", \"message\": \"🔥 义父早安！v5完整版报告：\\n\\n${RESULT}\"}" 2>/dev/null

echo "$(date): v5 选股完成" >> /Users/hsh/AIGC/AlphaGPT/daily_pick.log
