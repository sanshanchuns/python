from flask import request, json

user_info = {'name': 'letian'}
headers = {'content-type': 'application/json'}
r = request.post("http://127.0.0.1:5000/json", data=json.dumps(user_info), headers=headers)
print(r.headers)
print(r.json())