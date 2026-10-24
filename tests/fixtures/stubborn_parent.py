import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

if sys.argv[3] == "True":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
while not Path(sys.argv[2]).exists():
    time.sleep(0.01)
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"id": request["id"], "result": {"child": child.pid, "group": os.getpgrp()}}), flush=True)
while True:
    time.sleep(60)
