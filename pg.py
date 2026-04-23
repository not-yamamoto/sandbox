import os
import shlex
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm


# =========================
# 対象テーブルをここに書く
# schema.table 形式
# =========================
TABLES = [
    "aaa.bbb",
    "ccc.ddd",
    # "public.sample_table",
]

# 復元前に対象テーブルをDROPしたい場合は True
USE_CLEAN = True

# --if-exists を付けるか
USE_IF_EXISTS = True


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"環境変数 {name} が未設定です")
    return value


def run_command(cmd: list[str], env: dict[str, str], label: str) -> None:
    """
    コマンド実行。失敗時は stdout/stderr を含めて例外化。
    """
    result = subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"[{label}] 失敗\n"
            f"cmd: {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def build_pg_env(password: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    return env


def dump_table(
    table_name: str,
    dump_dir: Path,
    src_host: str,
    src_port: str,
    src_db: str,
    src_user: str,
    src_password: str,
) -> Path:
    dump_file = dump_dir / f"{table_name.replace('.', '__')}.dump"

    cmd = [
        "pg_dump",
        "-h", src_host,
        "-p", src_port,
        "-U", src_user,
        "-d", src_db,
        "-Fc",                    # custom format
        "-f", str(dump_file),
        "--table", table_name,
        "--no-owner",
        "--no-privileges",
    ]

    run_command(
        cmd,
        env=build_pg_env(src_password),
        label=f"dump {table_name}",
    )
    return dump_file


def restore_table(
    table_name: str,
    dump_file: Path,
    dst_host: str,
    dst_port: str,
    dst_db: str,
    dst_user: str,
    dst_password: str,
) -> None:
    cmd = [
        "pg_restore",
        "-h", dst_host,
        "-p", dst_port,
        "-U", dst_user,
        "-d", dst_db,
        "--no-owner",
        "--no-privileges",
    ]

    if USE_CLEAN:
        cmd.append("--clean")
    if USE_IF_EXISTS:
        cmd.append("--if-exists")

    cmd.append(str(dump_file))

    run_command(
        cmd,
        env=build_pg_env(dst_password),
        label=f"restore {table_name}",
    )


def main() -> None:
    load_dotenv()

    src_host = require_env("SRC_HOST")
    src_port = require_env("SRC_PORT")
    src_db = require_env("SRC_DB")
    src_user = require_env("SRC_USER")
    src_password = require_env("SRC_PASSWORD")

    dst_host = require_env("DST_HOST")
    dst_port = require_env("DST_PORT")
    dst_db = require_env("DST_DB")
    dst_user = require_env("DST_USER")
    dst_password = require_env("DST_PASSWORD")

    dump_dir = Path(os.getenv("DUMP_DIR", "./dump_files"))
    dump_dir.mkdir(parents=True, exist_ok=True)

    if not TABLES:
        raise ValueError("TABLES が空です")

    # 1テーブルずつ dump → restore
    for table_name in tqdm(TABLES, desc="tables", unit="table"):
        dump_file = dump_table(
            table_name=table_name,
            dump_dir=dump_dir,
            src_host=src_host,
            src_port=src_port,
            src_db=src_db,
            src_user=src_user,
            src_password=src_password,
        )

        restore_table(
            table_name=table_name,
            dump_file=dump_file,
            dst_host=dst_host,
            dst_port=dst_port,
            dst_db=dst_db,
            dst_user=dst_user,
            dst_password=dst_password,
        )

    print("完了しました")


if __name__ == "__main__":
    main()