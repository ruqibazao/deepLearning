# 切片
list_a = ["a","b","c","d","f"]
list_b = list_a[-1::-1] # end 元素不会包括 
print(list_b)
# ["b","d"]
list_c = list_a[1::2]
print(list_c)

# 所有的序列元素 字符串, 列表, 元组 都可以切片
text = "所有的序列元素"
# 字符串反转 
print(text[::-1])