# Version: V2.0
# detail:  training the import data and use the same data to predict vehicle state.
# data: 2022-03-29

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re

def parser_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_file', action='store', dest='origin_file' \
                        , default="None", help='planner log file')
    global planner_opts
    planner_opts = parser.parse_args()

class Planner_Data(object):
    def __init__(self):
        self.frequency = 10
        self.can_state_full = []
        self.can_cmd = []
        return

    def parse_cancmd(self, ori_file):
        pattern = re.compile(r'^can_cmd +')
        with open(ori_file, 'r') as log:
                self.can_cmd = []
                line = log.readline()
                while line:
                    match = pattern.match(line)
                    if match:
                        line_split = line.split(' : ')
                        line_split = line_split[1].split(',')
                        data_split = line_split[0].split('vel: ')
                        data_split = data_split[1].split(')')
                        vel = float(data_split[0])
                        data_split = line_split[1].split('acc: ')
                        data_split = data_split[1].split(')')
                        acc = float(data_split[0])
                        data_split = line_split[2].split('steer: ')
                        data_split = data_split[1].split(')')
                        steer = float(data_split[0])
                        data_split = line_split[5].split('brake: ')
                        data_split = data_split[1].split(')')
                        brake = float(data_split[0])
                        self.can_cmd.append([vel, steer, acc, brake])
                    line = log.readline()
        self.can_cmd = np.array(self.can_cmd)

    def parse_canstate_full(self, ori_file):
        pattern = re.compile(r'^can_state_full +')
        with open(ori_file, 'r') as log:
                self.can_state_full = []
                line = log.readline()
                while line:
                    match = pattern.match(line)
                    if match:
                        line_split = line.split('vel: ')
                        line_split = line_split[1].split(')')
                        vel = float(line_split[0])

                        line_split = line.split('steer: ')
                        line_split = line_split[1].split(')')
                        steer = float(line_split[0])

                        self.can_state_full.append([vel, steer])
                    line = log.readline()
        self.can_state_full = np.array(self.can_state_full)

training_data = Planner_Data()

class Neural_Network(object):
    def __init__(self):
        self.N = 920
        self.D_in = 40
        self.H = 1024
        self.D_out = 1
        self.learning_rate = 0.001
        self.training_steps = 30000
        self.xx = []
        self.yy = []
        return

    def creat_tensors_of_input_and_output(self, can_state_full, can_cmd):
        for i in range(self.N):
            x = []
            for j in range(self.D_in / 2):
                x.append(can_state_full[i + j, 0])
            for j in range(self.D_in / 2):
                x.append(can_cmd[i + j, 2])
            self.xx.append(x)
            self.yy.append([can_state_full[i + j + 1, 0]])

        self.xx = torch.Tensor(self.xx)
        self.yy = torch.Tensor(self.yy)
        #self.x = torch.rand(self.N, self.D_in)
        #self.y = torch.rand(self.N, self.D_out)
    def define_mode(self):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.D_in, self.H),
            torch.nn.ReLU(),
            torch.nn.Linear(self.H, self.D_out),
        )

    def define_loss_fun(self):
        self.lose_fn = torch.nn.MSELoss(reduction='sum')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)

    def run_training_steps(self):
        for t in range(self.training_steps):
            y_pred = self.model(self.xx)
            loss = self.lose_fn(y_pred, self.yy)

            if t % 100 == 0:
                print(t, loss.data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    def generate_pred_result(self, input_date):
        return self.model(torch.Tensor(input_date))

neural_network = Neural_Network()

def plot_can():

    # 1. get file dir of training data ->planner_opts
    parser_options()

    # 2. parse data
    planner_data.parse_cancmd(planner_opts.origin_file)
    planner_data.parse_canstate_full(planner_opts.origin_file)

    # 3.training model
    neural_network.creat_tensors_of_input_and_output(planner_data.can_state_full, planner_data.can_cmd)
    neural_network.define_mode()
    neural_network.define_loss_fun()
    neural_network.run_training_steps()

    # 4. pred
    y = [0.0] * 20
    for i in range(920):
        x = []
        for j in range(40 / 2):
            x.append(planner_data.can_state_full[i + j, 0])
        for j in range(40 / 2):
            x.append(planner_data.can_cmd[i + j, 2])

        y.append(neural_network.generate_pred_result(x))

    pred_state = []
    for i in range(20):
        pred_state.append(planner_data.can_state_full[i, 0])
    print(pred_state)
    for i in (range(200)):
        x = []
        for j in range(40 / 2):
            x.append(pred_state[i + j])
        for j in range(40 / 2):
            x.append(planner_data.can_cmd[i + j, 2])

        pred_state.append(neural_network.generate_pred_result(x))

    size = min(len(planner_data.can_state_full), len(planner_data.can_cmd))
    frame = np.arange(0, size)
    print(size)
    plt.title('Acc')
    plt.plot(frame, planner_data.can_state_full[0:size, 0], 'b', label="veh_vel")
    plt.plot(frame, planner_data.can_cmd[0:size, 2], 'r', label="expt_acc")
    plt.plot(frame[0: 940], y, 'x', label="veh_vel-pred_one_period")
    plt.plot(frame[0:220], pred_state[0:220], '.', label="veh_vel-pred_continue")

    plt.grid(True)
    plt.legend(loc="best")
    plt.xlabel('frame(100ms)')
    plt.ylabel('Vel(m/s), Acc(m/ss)')
    plt.show()

def main():
    plot_can()

if __name__ == '__main__':
    main()


