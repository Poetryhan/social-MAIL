import numpy as np

# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self, scenario_test, training_label):
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self, world, scenario_test, training_label):
        raise NotImplementedError()
