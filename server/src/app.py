import warnings
from flask import Flask
from flask_restful import Resource, Api
import time
import sys

from sentry_sdk.api import flush
from mdp import HexGridMDP, ValueIteration, HexEnvironment, Agent, environment_from_dict
from solvers import MaxCausalEntIRL, SimpleIRL
from games import PredatorGame
import numpy as np
from numpy.random import RandomState
from flask_restful import reqparse
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
np.random.seed(123)

sentry_sdk.init(
    "https://858581c7c8234dd393241da13da4f94b@sentry.io/1340805",
    integrations=[FlaskIntegration()]
)

# cred = credentials.Certificate(r'C:\Users\Toby\Downloads\onlinetesting-96dd3-firebase-adminsdk-mkici-5bbc1544df.json') 
# firebase_admin.initialize_app(cred)

firebase_admin.initialize_app()

db = firestore.client()

firebase_ref = db.collection(u'avoidanceIRL').document(u'pilot2').collection('subjects')

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

traj_parser = reqparse.RequestParser()
traj_parser.add_argument('game_ref')
traj_parser.add_argument('env')
traj_parser.add_argument('predatorPos')
traj_parser.add_argument('preyPos')
traj_parser.add_argument('nMoves')

env_parser = reqparse.RequestParser()
env_parser.add_argument('env')
env_parser.add_argument('game_ref')

game_parser = reqparse.RequestParser()
game_parser.add_argument('game_ref')


############################
# SET UP TASK ENVIORNMENTS
############################

# Get environments

games = {}

database_games = db.collection(u'avoidanceIRL_gameinfo').get()

for db_game in database_games:
    
    game_info = db_game.to_dict()

    # if it's not empty
    if 'game_type' in game_info:
        envs = [environment_from_dict(env) for env in game_info['environments']]
        game_info['environments'] = envs
        if 'n_envs' in game_info:
            game_info.pop('n_envs')
        game_info['game_reference'] = db_game.id
        game = PredatorGame(**game_info)
        games[db_game.id] = game

print(games)
print("CREATED ENVIRONMENTS", flush=True)


###########################################
# ENVIRONMENT / GAME / MOVEMENT RESOURCES #
###########################################

class GetTrajectory(Resource):

    def post(self):
        args = traj_parser.parse_args()
        print(args)

        games[args['game_ref']].environments[int(args['env'])].move_agent(0, int(args['predatorPos']))
        games[args['game_ref']].environments[int(args['env'])].move_agent(1, int(args['preyPos']))
        games[args['game_ref']].environments[int(args['env'])].fit_agent(0, method='numba')

        warnings.warn(str(args['preyPos']))

        trajectory = [int(i) for i in games[args['game_ref']].environments[int(args['env'])].agents[0].generate_trajectory(n_steps=int(args['nMoves']))[1:]]
        # print(str(trajectory))
        # warnings.warn(str(trajectory))
        games[args['game_ref']].environments[int(args['env'])].move_agent(0, games[args['game_ref']].environments[int(args['env'])].agents[0].startingPosition)
        games[args['game_ref']].environments[int(args['env'])].move_agent(1, games[args['game_ref']].environments[int(args['env'])].agents[1].startingPosition)

        return {'trajectory': trajectory}

class GetEnvironment(Resource):

    def post(self):

        args = env_parser.parse_args()
        env = int(args['env'])
        out = games[args['game_ref']].environments[int(env)].to_dict(['Trees', 'Dirt', 'Reward'], format='coords')
        out['features']['Predator_1'] = out['agents']['Predator_1']['position']
        out['features']['Prey_1'] = out['agents']['Prey_1']['position']
        out['predator_reward_function'] = {}
        out['predator_reward_function']['red'] = game.environments[0].to_dict()['agents']['Predator_1']['reward_function'][0]
        out['predator_reward_function']['trees'] = game.environments[0].to_dict()['agents']['Predator_1']['reward_function'][1]
        out['predator_reward_function']['robot'] = game.environments[0].to_dict()['agents']['Predator_1']['reward_function'][-1]
        out.pop('agents')

        print(out, flush=True)
        return out

class GetGameInfo(Resource):

    def post(self):

        args = env_parser.parse_args()
        out = games[args['game_ref']].to_dict()
        out.pop('environments')

        print(out, flush=True)
        return out

##############################
# SET UP SUBJECT ON FIREBASE #
##############################

subject_info_parser = reqparse.RequestParser()
subject_info_parser.add_argument('subjectID')
subject_info_parser.add_argument('game_ref')
subject_info_parser.add_argument('game_type')
subject_info_parser.add_argument('time')
subject_info_parser.add_argument('date')

class createSubjectData(Resource):

    def post(self):
        args = subject_info_parser.parse_args()

        task_ref = db.collection(u'avoidanceIRL').document(args['game_ref'])
        doc = task_ref.get()
        if not doc.exists:
            task_ref.set({}, merge=True)

        firebase_ref = db.collection(u'avoidanceIRL').document(args['game_ref']).collection('subjects')
        doc_ref = firebase_ref.document(args['subjectID'])
        doc_ref.set({
            u'subjectID': args['subjectID'],
            u'time': args['time'],
            u'date': args['date'],
            u'game_ref': args['game_ref'],
            u'game_type': args['game_type'],
            u'behaviourData': dict([('env_{0}'.format(i+1), {'predator': [], 'prey': []}) for i in range(len(games[args['game_ref']].environments))]),
            u'ratingData': dict([('env_{0}'.format(i+1), {'trees': -999, 'red': -999, 'prey': -999}) for i in range(len(games[args['game_ref']].environments))]),
            u'predictionData': dict([('env_{0}'.format(i+1), []) for i in range(len(games[args['game_ref']].environments))]),
            u'score': 0
        })

        return {'status': 1}

api.add_resource(createSubjectData, '/init')

#########################
# SAVE DATA TO FIREBASE #
#########################

data_parser = reqparse.RequestParser()
data_parser.add_argument('subjectID')
data_parser.add_argument('env')
data_parser.add_argument('game_ref')
data_parser.add_argument('game_type')
data_parser.add_argument('treesRating')
data_parser.add_argument('redRating')
data_parser.add_argument('preyRating')
data_parser.add_argument('preyMoves')
data_parser.add_argument('predatorMoves')
data_parser.add_argument('predictions')
data_parser.add_argument('score')


class SaveData(Resource):

    def post(self):
        args = data_parser.parse_args()

        firebase_ref = db.collection(u'avoidanceIRL').document(args['game_ref']).collection('subjects')
        doc_ref = firebase_ref.document(args['subjectID'])

        doc_ref.update({
            u'score': args['score']
        })
        print("SAVING", args)
        if args['treesRating'] is not None:
            doc_ref.update({
                u'ratingData.{0}.trees'.format(args['env']): int(args['treesRating']),
                u'ratingData.{0}.red'.format(args['env']): int(args['redRating']),
                u'ratingData.{0}.prey'.format(args['env']): int(args['preyRating'])
            })

        if len(args['predictions']):
            doc_ref.update({
                u'predictionData.{0}'.format(args['env']): [int(i) for i in args['predictions'].split(',')]
            })

        doc_ref.update({
            u'behaviourData.{0}.predator'.format(args['env']): [int(i) for i in args['predatorMoves'].split(',')],
            u'behaviourData.{0}.prey'.format(args['env']): [int(i) for i in args['preyMoves'].split(',')],
        })

        print(args, flush=True)


result_parser = reqparse.RequestParser()
result_parser.add_argument('subjectID')
result_parser.add_argument('game_ref')
result_parser.add_argument('game_type')
result_parser.add_argument('bonus_amount')

class GetResult(Resource):

    def post(self):

        args = result_parser.parse_args()

        firebase_ref = db.collection(u'avoidanceIRL').document(args['game_ref']).collection('subjects')
        doc_ref = firebase_ref.document(args['subjectID'])

        sub_data = doc_ref.get().to_dict()

        # Calculate correct predictions for learning
        if args['game_type'] == "training":
            all_predictions = []
            all_predator_moves = []

            for env in range(6):
                all_predictions += sub_data['predictionData']['env_{0}'.format(env+1)]
                all_predator_moves += sub_data['behaviourData']['env_{0}'.format(env+1)]['predator'][1:]

            all_predictions = np.array(all_predictions)
            all_predator_moves = np.array(all_predator_moves)

            if not(len(all_predictions) == len(all_predator_moves)):
                print('lengths differ')
                n_correct = 4

            else:
                selected = np.random.choice(range(len(all_predator_moves)), 5, replace=False)
                n_correct = (all_predictions[selected] == all_predator_moves[selected]).sum()
                n_correct = np.max([n_correct, 1])
        else:
            n_correct = 0

        score = int(sub_data['score'])

        payment = "{:.2f}".format(((score / 1000) * float(args['bonus_amount'])) + n_correct * 0.5)

        doc_ref.update({
            u'payment': payment
        })

        return {'score': str(score), 'n_correct': str(n_correct), 'payment': payment}

api.add_resource(GetTrajectory, '/trajectory')
api.add_resource(GetEnvironment, '/environment')
api.add_resource(GetGameInfo, '/gameinfo')
api.add_resource(SaveData, '/data')
api.add_resource(GetResult, '/result')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.2')
