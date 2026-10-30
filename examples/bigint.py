# Real MicroPython: functions, for/range, arbitrary-size integers and containers.
def factorial(n):
    value = 1
    for i in range(2, n + 1):
        value *= i
    return value

results = {"factorial100": factorial(100), "power": 2 ** 256}
print(results["factorial100"])
print(results["power"])
