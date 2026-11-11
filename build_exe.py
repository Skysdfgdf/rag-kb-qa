"""一键打包为 Windows 可执行程序：python build_exe.py

产物：dist/rag-kb-qa/rag-kb-qa.exe（onedir 模式，整个文件夹可拷贝迁移）
运行时数据（data/、.env、frontend/）都在 exe 同级目录。
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    print("[1/3] 运行 PyInstaller（首次打包需几分钟，请耐心等待）...")
    subprocess.check_call(
        [
            sys.executable, "-m", "PyInstaller",
            "--noconfirm", "--clean",
            "--onedir", "--console",
            "--name", "rag-kb-qa",
            "--hidden-import", "app",
            "--hidden-import", "app.main",
            "--collect-submodules", "uvicorn",
            # onnxruntime 新版本（>=1.22）在 PyInstaller 下会段错误，需 1.20.x
            # 并强制收集其全部 DLL；requirements.txt 已锁定版本。
            "--collect-all", "onnxruntime",
            # 默认 numpy 向量后端，不打包 chromadb（其 Rust 绑定打包后必崩，且体积大）
            "--exclude-module", "chromadb",
            "--exclude-module", "chromadb_rust_bindings",
            "--exclude-module", "chroma_hnswlib",
            str(ROOT / "launcher.py"),
        ],
        cwd=ROOT,
    )

    dist = ROOT / "dist" / "rag-kb-qa"
    print("[2/3] 复制前端与初始化目录 ...")
    shutil.copytree(ROOT / "frontend", dist / "frontend", dirs_exist_ok=True)
    (dist / "data").mkdir(exist_ok=True)

    print("[3/3] 打包完成！")
    print(f"  可执行程序: {dist / 'rag-kb-qa.exe'}")
    print("  提示: 把 .env 拷到 exe 同级目录可沿用原口令/模型配置；")
    print("        拷贝原 data/ 文件夹即可迁移已有知识库。")


if __name__ == "__main__":
    main()
