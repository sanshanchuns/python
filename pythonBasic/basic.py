import random
import matplotlib.pyplot as plt
import numpy as np

balance = 1000
threshold = 4
count = 0
history = []

while balance > 0:
    count += 1
    result = 0
    for i in range(3):
        result += random.choice(range(1, 7))

    prediction = random.choice(['押大', '押小'])
    if result >= 9 and prediction == '押大':
        balance += 5
    elif result < 9 and prediction == '押小':
        balance += 5
    else:
        balance -= 5

    print(str(count) + ' 次')
    print(balance)
    history.append(balance)

plt.plot(history)
plt.show()

for i in range(100):
    print(random.choice(["JGood", "is", "a", "handsome", "boy"]))

# class Solution:
#     def twoSum(self, nums, target):
#         """
#         :type nums: List[int]
#         :type target: int
#         :rtype: List[int]
#         """
#         # num_dict = {}
#         # for i, v in enumerate(nums):
#         #     rem = target - v
#         #     if rem in num_dict:
#         #         return [num_dict[rem], i]
#         #     num_dict[v] = i
#         #     print(num_dict)
#         # return None

#         num_list = []
#         for i, v in enumerate(nums):
#             rem = target - v
#             if rem in num_list:
#                 return [num_list.index(rem), i]
#             num_list.append(v)
#         return None

# s = Solution()
# print(s.twoSum([3, 5, 2, 4, 8, 1, 2, 777, 15], 6))
# print(list(v for i, v in enumerate([3, 2, 4, 15]) if v < 6))

# num_dict = {3: 0, 1: 1, 4: 2, 2: 3, -2: 4, 5: 5}

# l = [1, 5, 6, 2]
# if 5 in l:
#     print(l.index(5))


# class Solution:
#     def twoSum(self, nums, target):
#         """
#         :type nums: List[int]
#         :type target: int
#         :rtype: List[int]
#         """
#         num_dict = {}
#         for i, num in enumerate(nums):
#             rem = target - num
#             if rem in num_dict:
#                 return [num_dict[rem], i]
#             num_dict[num] = i
#         return None
