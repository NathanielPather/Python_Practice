result = ''
start = 1
for start in range(101):
    result += f'{start}: '
    if(start % 3 == 0):
        result += 'fizz'
    if (start % 5 == 0):
        result += 'buzz'
    result += '\n'
print(result)
