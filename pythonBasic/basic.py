class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # num_dict = {}
        # for i, v in enumerate(nums):
        #     rem = target - v
        #     if rem in num_dict:
        #         return [num_dict[rem], i]
        #     num_dict[v] = i
        #     print(num_dict)
        # return None

        num_list = []
        for i, v in enumerate(nums):
            rem = target - v
            if rem in num_list:
                return [num_list.index(rem), i]
            num_list.append(v)
        return None

s = Solution()
print(s.twoSum([3, 5, 2, 4, 8, 1, 2, 777, 15], 6))
# print(list(v for i, v in enumerate([3, 2, 4, 15]) if v < 6))

# num_dict = {3: 0, 1: 1, 4: 2, 2: 3, -2: 4, 5: 5}

l = [1, 5, 6, 2]
if 5 in l:
    print(l.index(5))


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
