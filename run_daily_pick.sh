#!/bin/bash
# AlphaGPT 每日选股定时任务

cd /Users/hsh/AIGC/AlphaGPT

# 设置环境
export PATH="/Users/hsh/Library/Python/3.9/bin:$PATH"
export TUSHARE_TOKEN="cc9f4227a4be5c67699791c24526d2ec3947877f1cec3619866078f4"

# 运行选股
/usr/bin/python3 /Users/hsh/AIGC/AlphaGPT/daily_pick.py >> /Users/hsh/AIGC/AlphaGPT/daily_pick.log 2>&1

# 记录完成时间
echo "====== $(date '+%Y-%m-%d %H:%M:%S') 选股完成 ======" >> /Users/hsh/AIGC/AlphaGPT/daily_pick.log

# 尝试通过 OpenClaw Gateway 发送通知
if curl -s -X POST "http://127.0.0.1:18789/api/message" \
  -H "Authorization: Bearer 06abb9963441f469fe6c5343accfc51e19486797afec1e51" \
  -H "Content-Type: application/json" \
  -d "{\"channel\": \"feishu\", \"target\": \"user:ou_c53ff42237108108087d63bdc539cf96\", \"message\": \"🚀 义父早安！今日选股报告已生成，请查看 daily_pick_result.txt\"}" 2>/dev/null; then
    echo "$(date): 通知已发送" >> /Users/hsh/AIGC/AlphaGPT/daily_pick.log
else
    echo "$(date): Gateway 未运行，通知发送失败" >> /Users/hsh/AIGC/AlphaGPT/daily_pick.log
fi
