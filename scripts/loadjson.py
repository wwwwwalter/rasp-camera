import json  
import os
import glob



# all_case_info_dict={}
# def load_case(directory):
#     global all_case_info
#     # 遍历case目录
#     for filename in glob.glob(os.path.join(directory, '*.json')): 
#         print(filename)
#         # 读取json文件
#         # case=0
#         with open(filename, 'r') as file: 
#             try: 
#                 case = json.load(file)
#             except:
#                 pass
#             else:
#                 key = case.keys()
#                 name=list(key)[0]
#                 all_case_info_dict[name]=case[name]

        






# load_case('case')
# print(all_case_info_dict.keys())
# print(all_case_info_dict['油耳'][0]['名称']) 

long_string = "这是一个非常长的字符串，需要被分割成多行进行显示。"  
  
# 假设你想要每10个字符就换行（这只是一个示例，你可能需要更复杂的逻辑）  
chunk_size = 10  
chunks = [long_string[i:i+chunk_size] for i in range(0, len(long_string), chunk_size)]  
  
# 使用join()和换行符来打印多行  
print("\n".join(chunks))


  
# print(data[0]['名称'])  
# print(data[0]['检查结果'])  
# print(data[0]['病因'])  
# print(data[0]['并发症'])  
# print(data[0]['简化版结果'])
# print(data[0]['简化版建议'])



# # 创建一个字典  
# my_dict = {  
#     'key1': 'value1',  
#     'key2': 'value2',  
#     'key3': 'value3'  
# }

# print(my_dict)
  
# # 获取所有的键  
# keys = my_dict.keys()  
  
# # 打印所有的键  
# print(keys)  # 输出: dict_keys(['key1', 'key2', 'key3'])  
  
# # 如果需要将这些键转换为列表  
# keys_list = list(keys)  
# print(keys_list)  # 输出: ['key1', 'key2', 'key3']  
  
# # 你可以直接迭代这些键  
# for key in my_dict.keys():  
#     print(key)  # 输出: key1, key2, key3（每个键分别打印）