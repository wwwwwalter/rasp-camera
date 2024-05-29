my_dict = {'a': 1, 'b': 2, 'c': 3}  
  
# 使用 get() 方法查找键 'd'，如果不存在则返回 None  
value = my_dict.get('d')  
print(value)  # 输出：None  
  
# 使用 get() 方法查找键 'd'，如果不存在则返回默认值 0  
value = my_dict.get('d', 0)  
print(value)  # 输出：0