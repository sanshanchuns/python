for i in range(10):
    print('{}'.format(i), end='')
print()

colors = ['red', 'green', 'blue', 'yellow']

for color in colors:
    print('{} '.format(color), end='')
print()

for color in reversed(colors):
    print('%s ' % color, end='')
print()

for i, color in enumerate(colors):
    print(i, color)

names = ['raymod', 'rachel', 'matthew']
for name, color in zip(names, colors):
    print(name, color)

for color in sorted(colors):
    print('{} '.format(color), end='')
print()

for color in sorted(colors, reverse=True):
    print('{} '.format(color), end='')
print()

for color in sorted(colors, key=len):
    print('{} '.format(color), end='')
print()

def find(seq, target):
    for i, value in enumerate(seq):
        if value == target:
            break
    else:
        return -1
    return i

print(find(range(10), 20))

d = {'matthew': 'blue', 'rachel': 'green', 'raymond': 'red'}

for k, v in d.items():
    print(k, v)



