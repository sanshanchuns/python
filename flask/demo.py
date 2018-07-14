from flask import Flask, request, url_for, render_template
from models import User

app = Flask(__name__)

@app.route('/')
def hello_world():
    content = 'hello world'
    return render_template('index.html', content=content)

@app.route('/user')
def hello_user():
    user = User(1, '灌篮高手')
    return render_template('user.html', user=user)

@app.route('/query_user/<user_id>')
def query_user(user_id):
    user = None
    if user_id == '1':
        user = User(1, '灌篮高手')
    return render_template('user.html', user=user)

@app.route('/users')
def show_users():
    users = []
    for i in range(1, 11):
        users.append(User(i, '灌篮高手'+ str(i)))
    return render_template('user.html', users=users)

@app.route('/base')
def base():
    return render_template('base.html')

@app.route('/one_base')
def one_base():
    return render_template('one_base.html')

@app.route('/two_base')
def two_base():
    return render_template('two_base.html')

# @app.route('/user', methods=['POST'])
# def hello_user():
#     return 'Hello User'

# @app.route('/user/<int:user_id>')
# def user_id(user_id):
#     return 'hello user ' + str(user_id)

# @app.route('/users/<user_id>')
# def user_id2(user_id):
#     return 'hello user2 ' + user_id

# @app.route('/query_user')
# def query_user1():
#     return 'hello ' + request.args.get('id')

# @app.route('/query_url')
# def query_url():
#     return 'query_url: ' + url_for('user_id', user_id=1)



if __name__ == '__main__':
    app.debug = True
    app.env = 'development'
    app.run()