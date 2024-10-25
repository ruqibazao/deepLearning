# 定义一个函数
def add(x,y):
    print(x+y)


# 定义一个函数
def test():
    print("好好学习, 天天向上")


if __name__=='__main__':
    print(123)
    # 当别人引用该模块的时候, 这个值已经是 包名+模块名称
    # 如果是自己直接运行, 就是__main__
    print(__name__) 
    print('abc')
