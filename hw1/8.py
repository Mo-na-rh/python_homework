# functions for recursion
# numbers of Fibonacchi

n = int(input("n = "))
def fib(n):
    if n in (1, 2):
        return 1
    return fib(n - 1) + fib(n - 2)
 
 
print(fib(n))

