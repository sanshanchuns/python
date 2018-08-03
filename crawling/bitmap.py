from bitarray import bitarray
import mmh3
from pybloomfilter import BloomFilter


# def bloomFilter(url):

#     offset = 2**31 - 1
#     bit_array = bitarray(4*1024*1024*1024)  #4G
#     bit_array.setall(0)

#     # mmh3 hash value 32 bit signed int
#     # add offset to make it unsinged int 0 ~  2^31-1

#     b1 = mmh3.hash(url, 42) + offset
#     bit_array[b1] = 1
# -*- coding:utf-8 -*-
 
import os
import sys 
import random
 
from pybloomfilter import BloomFilter
 
# 创建一个capacity等于100万，error rate等于0.001的bloomfilter对象
bfilter = BloomFilter(1000000,0.001,'bf_test.bloom')
 
# 添加100个元素
for x in range(1000000):
    bfilter.add(str(x))
 
# 与nmap文件同步
bfilter.sync()
 
 
# 测试error rate
error_in = 0
for x in range(2000000):
    if str(x) in bfilter and x > 1000000:
        error_in += 1
 
print("error_rate:%s" % (error_in*1.0/1000000))
    
