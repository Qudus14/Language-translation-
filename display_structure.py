import os

def display_structure(root_dir, indent=""):
    files_and_dirs = os.listdir(root_dir)
    files_and_dirs.sort()  # Sort for consistent output
    for item in files_and_dirs:
        path = os.path.join(root_dir, item)
        if os.path.isdir(path):
            print(f"{indent}{item}/")
            display_structure(path, indent + "    ")
        else:
            print(f"{indent}{item}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    display_structure(project_root)
