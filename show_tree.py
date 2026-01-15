import os


def list_files(startpath):
    # 要忽略的文件夹
    ignore_dirs = {'.git', '__pycache__', '.ipynb_checkpoints', '.idea', '.vscode'}

    print(f"{os.path.basename(os.path.abspath(startpath))}/")

    for root, dirs, files in os.walk(startpath):
        # 修改 dirs 列表以通过引用移除要忽略的目录
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * (level)
        print(f'{indent}├── {os.path.basename(root)}/')
        subindent = '│   ' * (level + 1)
        for f in files:
            # 过滤掉一些无关紧要的文件类型
            if not f.endswith('.pyc'):
                print(f'{subindent}└── {f}')


if __name__ == '__main__':
    # 打印当前目录结构
    list_files('.')