from flask import Flask
from flask_restful import Resource, Api
import time
import sys
from mdp import HexGridMDP, ValueIteration, HexEnvironment, Agent
from solvers import MaxCausalEntIRL, SimpleIRL
import numpy as np
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


firebase_admin.initialize_app()

db = firestore.client()

firebase_ref = db.collection(u'avoidanceIRL').document(u'pilot1').collection('subjects')

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

traj_parser = reqparse.RequestParser()
traj_parser.add_argument('env')
traj_parser.add_argument('predatorPos')
traj_parser.add_argument('preyPos')

env_parser = reqparse.RequestParser()
env_parser.add_argument('env')



############################
# SET UP TASK ENVIRONMENTS #
############################

# Define environments
environments = dict()

# ENVIRONMENT 1
features = np.zeros((3, 21 * 10))
features[1, np.random.randint(0, 21 * 10, 100)] = 1
features[0, 135:139] = 1
features[0, 145:139] = 1
features[0, 155:159] = 1

# Reward
features[2, np.random.random_integers(0, 209, 15)] = 1

testMDP1 = HexGridMDP(features, (21, 10))
testAgent1 = Agent('Predator_1', [1, 0, 0, 0, 0], position=200, solver_kwargs={'discount': 0.9, 'tol': 1e-4})
testAgent2 = Agent('Prey_1', [0, 0, 1, 0, 0], position=108, solver_kwargs={'discount': 0.9, 'tol': 1e-4})

testEnvironment1 = HexEnvironment(testMDP1, [testAgent1, testAgent2])
environments['env_0'] = testEnvironment1

# ENVIRONMENT 2
features = np.zeros((3, 21 * 10))
features[1, np.random.randint(0, 21 * 10, 100)] = 1
features[0, 35:40] = 1
features[0, 45:50] = 1
features[0, 55:60] = 1

# Reward
features[2, np.random.random_integers(0, 209, 15)] = 1
testAgent1 = Agent('Predator_1', [1, 0, 0, 0, 0], position=180, solver_kwargs={'discount': 0.9, 'tol': 1e-4})
testAgent2 = Agent('Prey_1', [0, 0, 1, 0, 0], position=108, solver_kwargs={'discount': 0.9, 'tol': 1e-4})

testMDP2 = HexGridMDP(features, (21, 10))
testEnvironment2 = HexEnvironment(testMDP2, [testAgent1, testAgent2])
environments['env_1'] = testEnvironment2

# ENVIRONMENT 3
features = np.zeros((3, 21 * 10))
features[1, np.random.randint(0, 21 * 10, 100)] = 1
for i in range(7):
    features[0, i*10:i*10+4] = 1

# Reward
features[2, np.random.random_integers(0, 209, 15)] = 1
testAgent1 = Agent('Predator_1', [1, 0, 0, 0, 0], position=209, solver_kwargs={'discount': 0.9, 'tol': 1e-4})
testAgent2 = Agent('Prey_1', [0, 0, 1, 0, 0], position=36, solver_kwargs={'discount': 0.9, 'tol': 1e-4})

testMDP3 = HexGridMDP(features, (21, 10))
testEnvironment3 = HexEnvironment(testMDP3, [testAgent1, testAgent2])
environments['env_2'] = testEnvironment3

# ENVIRONMENT 4
features = np.zeros((3, 21 * 10))
features[1, np.random.randint(0, 21 * 10, 100)] = 1
for i in range(8, 14):
    features[0, i*10+4:i*10+8] = 1
for i in range(5):
    features[0, i*10:i*10+3] = 1
for i in range(15, 20):
    features[0, i*10:i*10+3] = 1
for i in range(19, 21):
    features[0, i*10+5:i*10+10] = 1

# Reward
features[2, np.random.random_integers(0, 209, 15)] = 1
testAgent1 = Agent('Predator_1', [1, 0, 0, 0, 0], position=125, solver_kwargs={'discount': 0.9, 'tol': 1e-4})
testAgent2 = Agent('Prey_1', [0, 0, 1, 0, 0], position=36, solver_kwargs={'discount': 0.9, 'tol': 1e-4})

testMDP4 = HexGridMDP(features, (21, 10))
testEnvironment4 = HexEnvironment(testMDP4, [testAgent1, testAgent2])
environments['env_3'] = testEnvironment4

# ENVIRONMENT 5
features = np.zeros((3, 21 * 10))
features[1, np.random.randint(0, 21 * 10, 100)] = 1

for i in range(6):
    features[0, i*10:i*10+3] = 1

for i in range(8):
    features[2, i*10 + np.random.random_integers(3, 7, 3)] = 1

# Reward
# features[2, np.random.random_integers(0, 209, 15)] = 1
testAgent1 = Agent('Predator_1', [1, 0, 0, 0, 0], position=30, solver_kwargs={'discount': 0.9, 'tol': 1e-4})
testAgent2 = Agent('Prey_1', [0, 0, 1, 0, 0], position=36, solver_kwargs={'discount': 0.9, 'tol': 1e-4})

testMDP5 = HexGridMDP(features, (21, 10))
testEnvironment5 = HexEnvironment(testMDP5, [testAgent1, testAgent2])
environments['env_4'] = testEnvironment5

# ENVIRONMENT 6
features = np.zeros((3, 21 * 10))
features[1, np.random.randint(0, 21 * 10, 100)] = 1
features[1, 100:110] = 1

for i in range(7, 14):
    features[0, i*10+4:i*10+7] = 1

# Reward
features[2, np.random.random_integers(0, 209, 15)] = 1
testAgent1 = Agent('Predator_1', [1, 0, 0, 0, 0], position=100, solver_kwargs={'discount': 0.9, 'tol': 1e-4})
testAgent2 = Agent('Prey_1', [0, 0, 1, 0, 0], position=115, solver_kwargs={'discount': 0.9, 'tol': 1e-4})

testMDP6 = HexGridMDP(features, (21, 10))
testEnvironment6 = HexEnvironment(testMDP6, [testAgent1, testAgent2])
environments['env_5'] = testEnvironment6




class GetTrajectory(Resource):

    def post(self):
        args = traj_parser.parse_args()

        environments[args['env']].move_agent(0, int(args['predatorPos']))
        environments[args['env']].move_agent(1, int(args['preyPos']))

        environments[args['env']].fit_agent(0, method='numba')
        trajectory = [int(i) for i in environments[args['env']].agents[0].generate_trajectory(n_steps=2)[1:]]

        environments[args['env']].move_agent(0, environments[args['env']].agents[0].startingPosition)
        environments[args['env']].move_agent(1, environments[args['env']].agents[1].startingPosition)

        return {'trajectory': trajectory}

class GetEnvironments(Resource):

    def post(self):
        args = env_parser.parse_args()
        env = args['env']
        out = environments[env].to_dict(['Trees', 'Dirt', 'Reward'])

        print(out, flush=True)
        return out


##############################
# SET UP SUBJECT ON FIREBASE #
##############################

subject_info_parser = reqparse.RequestParser()
subject_info_parser.add_argument('subjectID')
subject_info_parser.add_argument('time')
subject_info_parser.add_argument('date')

class createSubjectData(Resource):

    def post(self):
        args = subject_info_parser.parse_args()
        doc_ref = firebase_ref.document(args['subjectID'])
        doc_ref.set({
            u'subjectID': args['subjectID'],
            u'time': args['time'],
            u'date': args['date'],
            u'behaviourData': dict([('env_{0}'.format(i+1), {'predator': [], 'prey': []}) for i in range(len(environments))]),
            u'ratingData': dict([('env_{0}'.format(i+1), {'trees': -999, 'red': -999, 'prey': -999}) for i in range(len(environments))]),
            u'predictionData': dict([('env_{0}'.format(i+1), []) for i in range(len(environments))]),
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

        doc_ref = firebase_ref.document(args['subjectID'])

        doc_ref.update({
            u'score': args['score']
        })

        doc_ref.update({
            u'ratingData.{0}.trees'.format(args['env']): int(args['treesRating']),
            u'ratingData.{0}.red'.format(args['env']): int(args['redRating']),
            u'ratingData.{0}.prey'.format(args['env']): int(args['preyRating'])
        })

        doc_ref.update({
            u'behaviourData.{0}.predator'.format(args['env']): [int(i) for i in args['predatorMoves'].split(',')],
            u'behaviourData.{0}.prey'.format(args['env']): [int(i) for i in args['preyMoves'].split(',')],
            u'predictionData.{0}'.format(args['env']): [int(i) for i in args['predictions'].split(',')]
        })

        print(args, flush=True)


result_parser = reqparse.RequestParser()
result_parser.add_argument('subjectID')

class GetResult(Resource):

    def post(self):

        args = result_parser.parse_args()

        doc_ref = firebase_ref.document(args['subjectID'])

        sub_data = doc_ref.get().to_dict()


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

        score = int(sub_data['score'])

        payment = "{:.2f}".format((score / 2000) + n_correct * 0.5)

        doc_ref.update({
            u'payment': payment
        })

        return {'score': str(score), 'n_correct': str(n_correct), 'payment': payment}

api.add_resource(GetTrajectory, '/trajectory')
api.add_resource(GetEnvironments, '/environment')
api.add_resource(SaveData, '/data')
api.add_resource(GetResult, '/result')

if __name__ == '__main__':
    app.run()