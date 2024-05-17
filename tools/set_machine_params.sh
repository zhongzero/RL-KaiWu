#!/bin/bash
# 安装KaiwuDRL时, 设置机器上的参数配置

# 设置/tmp文件删除规则, /tmp文件保留最近1天即可, crontab增加定时任务, 每隔1小时运行
echo "*/60 * * * * find /tmp -type f -atime +1 -delete;" >> /var/spool/cron/root