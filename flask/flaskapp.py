from flask import Flask, flash, render_template, request, Response, abort, json

app = Flask(__name__)
app.secret_key = '123'

@app.route('/')
def hello_world():
    flash('hello world')
    return render_template('index.html')

@app.route('/json', methods=['POST'])
def my_json():
    print(request.headers)
    print(request.json)

    form = request.form
    print(form.get('name'))

    # rt = {'info':'hello '+request.json['name']}
    # return Response(json.dumps(rt),  mimetype='application/json')

    return Response(json.dumps({'info': 'hello'}),  mimetype='application/json')

@app.route('/login', methods=['POST'])
def login():
    form = request.form
    username = form.get('username')
    password = form.get('password')
    
    if not username:
        flash('please enter username')
        return render_template('index.html')

    if not password:
        flash('please enter password')
        return render_template('index.html')
    
    if username == '123' and password == '456':
        flash('login success')
        return render_template('index.html')
    else:
        flash('username or password is wrong')
        return render_template('index.html')
 
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html')

@app.route('/users/<user_id>')
def user(user_id):
    if user_id == '1':
        user.user_name = 'test'
        return render_template('user.html', user=user)
    else:
        abort(404)

if __name__ == '__main__':
    app.debug = True
    app.env = 'development'
    app.run()