"""
1 定义一个学生类
2 在类中定义构造函数(初始化方法)
3 给类添加属性  对象属性, 类属性
4 给类添加 对象方法  
5 给类添加 类方法  静态方法
6 类也可以继承
"""

class Person:
    # 初始化方法: 魔法方法(双下划线开头, 双下划线结尾)
    # 魔法方法 需要我们自己显示调用, 在程序执行一定的时机, 会自动的调用函数
    def __init__(self, name, age):
        # self 当前对象 和java中的this一样
        # pass  # 占位符
        self.name = name
        self.age = age

    def intro(self):
        print(f"大家好, 我是{self.name},今年{self.age}")

    # 定义一个类方法
    @classmethod   
    def test(cls): # cls:class
        print(f"这是一个类方法:")

    # 静态方法
    @staticmethod
    def test2():
        print('这是一个静态方法')


# 创建一个对象  ① 先调用__new__方法 ② 再去调用初始化方法
p = Person("老王",18)
# 调用方法
p.intro()
# 可以直接使用对象进行调用
Person.test()
Person.test2()

print("="*50)

# 定义一个学生类 继承Person
class Student(Person):
    def __init__(self, name, age, score):
        self.score = score
        super().__init__(name, age)

    def intro(self):
        print(f"我的名字是:{self.name},今年{self.age}, 成绩{self.score}")

# 创建一个学生对象
s1 = Student("老李",21,100)
s1.intro()

