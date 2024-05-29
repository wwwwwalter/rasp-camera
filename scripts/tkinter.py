import tkinter as tk  
  
def create_additional_window(master, title):  
    top = tk.Toplevel(master)  
    top.title(title)  
    label = tk.Label(top, text="Hello from another window!")  
    label.pack()  
  
def create_main_window():  
    root = tk.Tk()  
    root.title("Main Window")  
    label = tk.Label(root, text="Hello, World!")  
    label.pack()  
      
    # 创建一个额外的窗口  
    create_additional_window(root, "Additional Window")  
      
    root.mainloop()  
  
# 创建主窗口  
create_main_window()