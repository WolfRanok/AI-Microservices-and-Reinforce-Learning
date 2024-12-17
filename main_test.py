a = 1
def fun():
    global a
    a = 21

print(a)
fun()
print(a)