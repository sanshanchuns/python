from google.protobuf import json_format

import addressbook_pb2

person = addressbook_pb2.Person()

person.id = 1
person.name = "leo"
person.email = "ace.leo.zhu@gmail.com"
person.phones.add(number="18888888888", type=addressbook_pb2.Person.MOBILE)

# 序列化成二进制Serialize
print("\nperson to bytes: ")
v_bytes = person.SerializeToString()
print(len(v_bytes))
print(v_bytes)

# 从二进制反序列化Parse
person1 = addressbook_pb2.Person()
person1.ParseFromString(person.SerializeToString())

# 转换成字典Dict
print("\nperson to dict: ")
v_dict = json_format.MessageToDict(person1, True)
print(v_dict)

# 从json数据反序列化Parse
person2 = addressbook_pb2.Person()
v_json = json_format.MessageToJson(person1, True)
print("\nperson to json: ")
print(len(v_json))
print(v_json)

json_format.Parse(v_json, person2)
print("\nperson from json: ")
print(person2)