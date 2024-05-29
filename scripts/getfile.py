# import os  
# import glob  
  
# # 获取当天保存图片目录中最新的一张照片的名称 
# def get_latest_jpg_file(directory):  
#     # 初始化最新文件的信息
#     latest_file = None
#     latest_filename_only = None  
#     latest_file_time = 0  
  
#     # 遍历指定目录下的所有.jpg文件  
#     for filename in glob.glob(os.path.join(directory, '*.jpg')):  
#         if os.path.isfile(filename):  # 确保是一个文件，而不是目录  
#             # 获取文件的修改时间  
#             file_time = os.path.getmtime(filename)  
  
#             # 如果当前文件的修改时间比已知的最新文件时间更新，则更新最新文件信息  
#             if file_time > latest_file_time:  
#                 latest_file_time = file_time  
#                 latest_file = filename
#                 latest_filename_only = os.path.basename(latest_file)  # 提取文件名
  
#     return latest_filename_only 
  
# # 使用示例  
# directory = 'output/2024/05/30'  # 替换为你的目录路径  
# latest_jpg_file = get_latest_jpg_file(directory)  
# if latest_jpg_file:  
#     print(f"{latest_jpg_file}")  
# else:  
#     print("No .jpg files found in the directory.")
    


import os  
  
def replace_extension(filename, new_extension):  
    # 使用os.path.splitext()分割文件名和扩展名  
    base_name, extension = os.path.splitext(filename)  
      
    # 确保新的扩展名以'.'开头，如果不是，则添加它  
    if not new_extension.startswith('.'):  
        new_extension = '.' + new_extension  
      
    # 重新组合文件名和新的扩展名  
    new_filename = base_name + new_extension  
      
    return new_filename  
  
# 使用函数  
old_filename = "example.txt"  
new_extension = 'py'  
new_filename = replace_extension(old_filename, new_extension)  
print(new_filename)  # 输出: example.py