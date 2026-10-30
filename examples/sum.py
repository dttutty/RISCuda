i = 1
total = 0
while i <= 137:
    if i % 3 == 0:
        total = total + i * i
    else:
        total = total - 2 * i
    i = i + 1
print(total)
