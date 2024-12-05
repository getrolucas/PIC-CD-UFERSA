import os

def config():
    """
    Ajusta o diretório do notebook.
    """
    global main_path
    try:
        main_path
    except NameError:
        main_path = os.getcwd()
    if os.getcwd() != main_path:
        os.chdir(main_path)

if __name__ == '__main__':
    for i in range(5):
        os.chdir("..")
        config()
        print(os.getcwd())
