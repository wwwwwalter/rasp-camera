import numpy as np  
  
# 假设我们有5种疾病的概率  
disease_probability = np.array([0.2, 0.5, 0.1, 0.7, 0.3])  
  
# 使用np.argsort()得到排序索引，并通过[::-1]反转得到从大到小的顺序  
disease_probability_indx = np.argsort(disease_probability)[::-1]  
  
# 打印排序后的索引  
print("疾病概率从高到低的排序索引：", disease_probability_indx)  
  
# 如果我们想看排序后的疾病概率值，可以这样做：  
sorted_disease_probability = disease_probability[disease_probability_indx]  
print("排序后的疾病概率：", sorted_disease_probability)