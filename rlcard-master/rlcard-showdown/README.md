# 演示验证系统
我们利用RLCard Showdown提供的框架构造我们的红十游戏GUI演示验证系统

## Installation
our system separated frontend and backend. The frontend is built with React and the backend is based on Django and Flask.

### Prerequisite
To set up the frontend, you should make sure you have [Node.js](https://nodejs.org/) and NPM installed. Normally you just need to manually install Node.js, and the NPM package would be automatically installed together with Node.js for you. Please refer to its official website for installation of Node.js.

You can run the following commands to verify the installation
```
node -v
npm -v
```
For backend, make sure that you have **Python 3.6+** and **pip** installed.

### Install Frontend and Backend
The frontend can be installed with the help of NPM:
```
npm install
```
The backend of leaderboard can be installed with
```
pip3 install -r requirements.txt
cd server
python3 manage.py migrate
cd ..
```

### Install Red_10 game Env
```
cd ..
pip install -e .
```


### Run 
1. Launch the backend of leaderboard with

we set the trained weights of relation network and dangerous network in rlcard-master/rlcard/models/pretrained/RD_pretrained

we set the trained four styles weights set of Douzero model in rlcard-master/rlcard/models/pretrained/red_10_pretrained

```
cd server
python3 manage.py runserver
```

2. Run the following command in another new terminal under the project folder to start frontend:
```
npm start
```
You can view leaderboard at [http://127.0.0.1:3000/](http://127.0.0.1:3000/) . The backend of leaderboard will run in [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

