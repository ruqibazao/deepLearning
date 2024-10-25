# 在Python中, 所谓的模块, 就是一个py文件
# a.py  --> 模块a
# hello.py  --> hello的模块
# test.py  ---> test的模块
# 包: 管理模块的一个目录结构

# 程序的入口
if __name__=='__main__':
    # 需要使用data模块(data.py)的add函数==> 导入尽量
    from test1 import data  # 导入模块
    # 调用函数  模块.函数()
    data.add(1,6)

    from test1.data import add  # 从test1包下面的data模块 导入add函数
    add(8,5)

    # 调用内置的模块
    # 导入数学模块
    import math

    # 调用方式: 模块名.函数名
    print(math.pi)
    print(math.sqrt(4)) # 平方根函数
    print(math.e)

    # 导入随机数模块
    from random import randint, shuffle
    print(randint(1,10))
    data = [1,2,3,4,5]
    shuffle(data)
    print(data)