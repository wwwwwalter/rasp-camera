
  

def sub():
    # 如果我们想要修改全局变量的值，我们需要在修改它之前使用 global 关键字  
    # global global_variable  # 声明我们要修改的是全局变量  
    global_variable = "I was sub"  
    print(global_variable)  # 输出: I am global 
  
def my_function():  
    # 在函数内部，我们可以直接访问全局变量，不需要使用 global 关键字  

      
    # 如果我们想要修改全局变量的值，我们需要在修改它之前使用 global 关键字  
    global global_variable  # 声明我们要修改的是全局变量  
    global_variable = "I was modified in the function"  
    print(global_variable)  # 输出: I am global  
    sub()
    


  

if __name__ == "__main__":
    # 在全局作用域中定义一个变量  
    global_variable = "I am global"
    my_function()  


