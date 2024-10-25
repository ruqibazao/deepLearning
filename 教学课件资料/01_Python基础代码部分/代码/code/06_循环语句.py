# for in range() 
# range() 是一个函数, 用于生成一系列的数据 并且返回一个可迭代对象(列表)
# range(start=1, end=100, step=1)
# start: 开始数据, 如果不写, 默认为0 
# end:   结束数据, 必须填写的参数, 生成的数据不会包括end位置
# step:  步长 生成的时候, 一次迈多少步, 如果不写, 默认为1或者-1

print(range(10))  # range(0,10,1)
for i in range(10):
    print(i)

print("="*50)
for i in range(10,2): # range(0,10,2)
    print(i)
print("="*50)
for i in range(0,10,2):
    print(i)