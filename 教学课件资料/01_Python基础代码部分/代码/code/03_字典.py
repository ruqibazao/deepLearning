# 定义一个字典
person= {"name":"lucy","age":18}
# 查看变量的数据类型 type
print(type(person))  # <class 'dict'>

# 增, 删, 改, 查
person["sex"]="man"

# 删除元素(根据key删除)
del person["sex"]

# 修改
person['age']=28
print(person)

# 查询 for
for item in person.items():
    print(f"{item[0]}:{item[1]}")    

print(f"=="*50)
# person.items()==> 元组 ("name","lucy") ===> 拆包 ===> key, value
for key,value in person.items():
    print(f"{key}:{value}")    
print(f"=="*50)
for key in person.keys():
    print(f"{key}:{person[key]}")

