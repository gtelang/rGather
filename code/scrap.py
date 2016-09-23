
print "Hello ", \
      "World"

print [i for i        \
         in range(10) \
         if i%2 == 0]

def f(a,
      b):
    return a+b

e = (2,3)
print f(*e)
