$ErrorActionPreference = 'Stop'
$dir = 'C:\Projects\Transcipt-main\tg-bot'
$dst = Join-Path $dir "backups\bot-$(Get-Date -Format yyyyMMdd-HHmm).db"
New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null

& 'C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe' -c "import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); t=sqlite3.connect(sys.argv[2]); s.backup(t); t.close(); s.close()" (Join-Path $dir 'bot.db') $dst

# держим копии за две недели
Get-ChildItem (Split-Path $dst) -Filter 'bot-*.db' |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-14) } |
    Remove-Item -Force
