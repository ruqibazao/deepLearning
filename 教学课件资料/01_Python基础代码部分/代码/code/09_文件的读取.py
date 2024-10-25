# 读文件
# 1 打开文件
f = open("01_hello.py",mode="r",encoding="utf-8")
# 2 读取数据
print(f.read())
# 3 关闭文件
f.close()

# 写文件
f2 = open("test.txt",mode="w",encoding="utf-8")
# 写数据
f2.write("好好听课, 别瞎搞")
f2.close()

with open("test2.txt", mode="w", encoding="utf-8") as f:
    f.write('12345678')
