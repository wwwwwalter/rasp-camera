# mymodule.py  
  
def my_function():  
    print("This is a function in mymodule.")  
    print(__name__)
  
if __name__ == "__main__":  
    print("mymodule is being run directly.")  
    my_function()