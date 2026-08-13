"""本机直连远程服务器执行命令（paramiko 密码认证，无交互，不经 WSL 中转）。

用法:
    python ssh_remote.py "<remote command>"
    python ssh_remote.py --sftp-put <local> <remote>     # 上传单个文件
    python ssh_remote.py --sftp-get <remote> <local>     # 下载单个文件
    python ssh_remote.py --sync <local_dir> <remote_dir> # 递归同步（排除本地 venv 大目录）

依赖: pip install paramiko
"""
import os
import sys

import paramiko

HOST = "172.18.12.5"
PORT = 22
USER = "xidian"
PASSWORD = "123!@#qwe"

# 同步时排除的本地目录/后缀（本地大环境目录不上传）
SYNC_EXCLUDE_DIRS = {
    "venv", ".venv", ".egg-venv", ".pylibs", "__pycache__", ".git", ".idea",
    ".vscode", "node_modules", "wandb", "outputs",
}
SYNC_EXCLUDE_EXTS = {".pyc", ".log"}


def _connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST, port=PORT, username=USER, password=PASSWORD,
        look_for_keys=False, allow_agent=False, timeout=20,
    )
    return client


def run(cmd, timeout=3600):
    """执行远端命令，返回 (exit_code, stdout, stderr)。"""
    client = _connect()
    try:
        stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()
    finally:
        client.close()
    return rc, out, err


def sftp_put(local, remote):
    client = _connect()
    try:
        sftp = client.open_sftp()
        sftp.put(local, remote)
        sftp.close()
        print(f"uploaded {local} -> {remote}")
    finally:
        client.close()


def sftp_get(remote, local):
    client = _connect()
    try:
        sftp = client.open_sftp()
        sftp.get(remote, local)
        sftp.close()
        print(f"downloaded {remote} -> {local}")
    finally:
        client.close()


def sftp_sync(local_dir, remote_dir):
    """递归上传本地目录到远端，跳过 SYNC_EXCLUDE_DIRS / SYNC_EXCLUDE_EXTS。"""
    local_dir = os.path.abspath(local_dir)
    client = _connect()
    try:
        sftp = client.open_sftp()
        n = 0
        total = 0
        for root, dirs, files in os.walk(local_dir):
            dirs[:] = [d for d in dirs if d not in SYNC_EXCLUDE_DIRS]
            rel = os.path.relpath(root, local_dir)
            rem = remote_dir if rel == "." else f"{remote_dir}/{rel.replace(os.sep, '/')}"
            try:
                sftp.stat(rem)
            except FileNotFoundError:
                sftp.mkdir(rem)
            for f in files:
                if os.path.splitext(f)[1] in SYNC_EXCLUDE_EXTS:
                    continue
                local_f = os.path.join(root, f)
                remote_f = f"{rem}/{f}"
                try:
                    st = sftp.stat(remote_f)
                    if st.st_size == os.path.getsize(local_f):
                        continue  # 大小一致则跳过（增量）
                except FileNotFoundError:
                    pass
                sftp.put(local_f, remote_f)
                n += 1
                total += os.path.getsize(local_f)
        sftp.close()
        print(f"synced {local_dir} -> {remote_dir}: {n} files, {total/1e6:.1f} MB")
    finally:
        client.close()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--sftp-put":
        sftp_put(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 3 and sys.argv[1] == "--sftp-get":
        sftp_get(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 3 and sys.argv[1] == "--sync":
        sftp_sync(sys.argv[2], sys.argv[3])
    else:
        rc, out, err = run(sys.argv[1])
        sys.stdout.write(out)
        if err.strip():
            sys.stderr.write("[stderr]\n" + err)
        sys.exit(rc)
