from google.protobuf import json_format

import user_pb2

user = user_pb2.User()

user.uid = 150
# user.name = "leo"

# 序列化成二进制Serialize
print("\nperson to bytes: ")
v_bytes = user.SerializeToString()
print(len(v_bytes))
print(v_bytes)

# # 转换成字典Dict
# print("\nperson to dict: ")
# v_dict = json_format.MessageToDict(user, True)
# print(v_dict)

# 从json数据反序列化Parse
v_json = json_format.MessageToJson(user, True)
print("\nperson to json: ")
print(len(v_json))
print(v_json)