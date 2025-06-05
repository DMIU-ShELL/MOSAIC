#######################################################################
# Copyright (C) 2017 Shangtong Zhang (zhangshangtong.cpp@gmail.com)   #
# Copyright (C) 2025 Saptarshi Nath, Christos Peridis,                #
# Eseoghene Benjamin, Andrea Soltoggio                                #
#                                                                     #
# Licensed under the Apache License, Version 2.0 (the "License");     #
# you may not use this file except in compliance with the License.    #
# You may obtain a copy of the License at                             #
#     http://www.apache.org/licenses/LICENSE-2.0                      #
#                                                                     #
# Unless required by applicable law or agreed to in writing, software #
# distributed under the License is distributed on an "AS IS" BASIS,   #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or     #
# implied. See the License for the specific language governing        #
# permissions and limitations under the License.                      #
#######################################################################

import numpy as np


class Replay:
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0

    def feed(self, experience):
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

    def feed_batch(self, experience):
        experience = zip(*experience)
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = sampled_data#list(map(lambda x: np.asarray(x)), zip(*sampled_data))
        return batch_data

    def size(self):
        return len(self.data)

    def empty(self):
        return not len(self.data)

    def clear(self):
        self.data.clear()
        self.pos = 0

'''class Replay:
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0

    def feed(self, experience):
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

    def feed_batch(self, experience):
        experience = zip(*experience)
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        return batch_data
    
    def size(self):
        return len(self.data)

    def empty(self):
        return not len(self.data)
    

class Replay2:
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0

    def feed(self, experience):
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

    def feed_batch(self, experience):
        experience = zip(*experience)
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = batch_data = sampled_data#list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        return batch_data

    def size(self):
        return len(self.data)

    def empty(self):
        return not len(self.data)

    def clear(self):
        self.data.clear()
        self.pos = 0'''