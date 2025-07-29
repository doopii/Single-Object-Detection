from tkinter import Tk
from ui import ObjectDetectorApp  

def main():
    root = Tk()
    app = ObjectDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
