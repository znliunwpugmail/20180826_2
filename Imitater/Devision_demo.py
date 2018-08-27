from DQN_net import dqn
import numpy as np
import random
import copy
EPISDOE = 10000
STEP = 10000
LAMBDA = 0.1

class Devision():
    def __init__(self):
        self.capacility_states = [500,500,10]
        self.states = []
        self.user_states = []
        self.es = []
        self.inouts = []
        self.id = []
        self.actions = []
        self.id_actions_dict = {}

    def get_states(self,filename="user_imitater_u_e_all_final.txt"):
        with open(filename) as f:
            line = f.readline()
            while line:
                content_str = line.split('\t')
                if(len(content_str)<6):
                    continue
                user_state = [float(content_str[3]),float(content_str[4]),float(content_str[5])]
                inout = int(content_str[2])
                e = [int(content_str[6])-int(content_str[7])]
                self.user_states.append(user_state)
                self.es.append(e)
                self.inouts.extend([inout])
                self.actions = []
                self.id.append(int(content_str[1]))
                line = f.readline()
        self.actions = np.ones(shape=(len(self.user_states)),dtype=np.int)*(-1)

    def relu(self,x):
        """Compute softmax values for each sets of scores in x."""
        x = np.array(x)
        x[x<0] = 0
        return x
    def train(self):
        state_index = 0
        action_dim = 3
        state_dim = action_dim * 2 + 1
        agent = dqn.DQN(action_dim, state_dim)
        in_num = 0
        out_num = 0

        while 1 == 1:
            total_reward = 0
            user_state = self.user_states[state_index]
            print(type(user_state))
            e_state = self.es[state_index]
            inout =self.inouts[state_index]
            print(self.inouts)
            if 0 == inout:
                print('inout',inout)
                print(state_index)
                self.capacility_states[self.actions[state_index]]+=1
                state_index+=1
                state_index=state_index%len(self.user_states)
                out_num+=1
                continue
            state = copy.deepcopy(user_state)
            state.extend(self.capacility_states)
            state.extend(e_state)
            in_num+=1
            print(state_index)
            for episode in range(EPISDOE):
                print('1',state_index)

                inout = self.inouts[state_index]
                if 0 == inout:
                    print('inout',inout)
                    print(self.capacility_states)
                    print(self.actions[state_index])
                    user_id = self.id[state_index]
                    print('state_index',state_index)
                    user_id_first_index = self.id.index(user_id)
                    print(user_id_first_index)
                    print(user_id)
                    print(self.id_actions_dict)
                    print(user_id_first_index)
                    print('fd', state_index)
                    if user_id in self.id_actions_dict.keys():
                        action_out = self.id_actions_dict[user_id]
                        print(action_out)
                        self.capacility_states[action_out[0]] += 1
                    state_index += 1
                    state_index = state_index % len(self.user_states)
                    print(self.capacility_states)
                    out_num += 1
                    continue

                action = agent.get_action([state])
                user_id = self.id[state_index]
                print('action', action)
                self.capacility_states[action[0]] -= 1
                self.id_actions_dict[user_id] = action
                print('2',state_index)
                print('in_out',in_num, out_num)
                next_state = copy.deepcopy(self.user_states[state_index])
                next_state.extend(self.capacility_states)
                next_state.extend(self.es[state_index])
                g_at = 1
                next_state = np.array(next_state)
                reward = g_at - LAMBDA * np.max(self.relu(-1*next_state[3:6]))
                print(reward)
                total_reward+=reward
                agent.percieve(state, action, reward, next_state, False)
                print(self.capacility_states)
                state = next_state.tolist()
                state_index += 1
                state_index = state_index % len(self.user_states)
                print('total_reward',total_reward)


                # for user_list in user_state_list:
                #     user_states.append(user_list[1:4])
                # capacility_state = self.capacility_create.capacility_state
                # print('start ')
                # for step in range(len(user_states)):
                #     user_state = user_states[step]
                #     state = user_state+capacility_state.tolist()
                #     action = agent.get_action([state])
                #     accept_propability = np.array(user_state)[action]
                #     rand_prop = random.uniform(0, 1)
                #     if rand_prop<accept_propability:
                #         g_at = 1
                #     else:
                #         g_at = 0
                #         # action = action_dim-1
                #     self.user_create.alluser_action.append(action)
                #     if step == len(user_states)-1:
                #         next_user_state = user_state
                #     else:
                #         next_user_state = user_states[step+1]
                #
                #     self.capacility_create.compute_decrease_capacility(action)
                #     next_capacility_state = self.capacility_create.capacility_state
                #     reward = g_at - LAMBDA * np.max(self.relu(-1*next_capacility_state)//500)
                #     total_reward+=reward
                #     next_state = next_user_state+next_capacility_state.tolist()
                #     agent.percieve(state,action,reward,next_state,False)
                #     print(capacility_state)
                #
                # print('total reward this episode is: ', total_reward)

if __name__ == '__main__':
    devision = Devision()
    devision.get_states()
    devision.train()


