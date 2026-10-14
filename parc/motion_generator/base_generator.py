"""
A kinematic motion generator class.
Generates a variable length sequence of motions, optionally conditioned on:
 - previous N character states
 - goal/objective
 - action ID
 - observations
"""

import abc


class MotionGenerator:
    def __init__(self, cfg):
        return
    
    @abc.abstractmethod
    def gen_sequence(self, cond):
        """
        cond: a dictionary of conditions
        """
        motion_seq = None
        info = None
        return motion_seq, info