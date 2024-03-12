#!/usr/bin/env python3

"""
Do a local practice grading.
The score you recieve here is not an actual score,
but gives you an idea on how prepared you are to submit to the autograder.
"""

import os
import sys

import pandas
import numpy
import sklearn.dummy

import autograder.question
import autograder.assignment
import autograder.style

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(THIS_DIR, 'data.txt')

class HO5(autograder.assignment.Assignment):
    def __init__(self, **kwargs):
        super().__init__(
            name = 'Practice Grading for Hands-On 5',
            additional_data = {
                'data': pandas.read_csv(DATA_PATH, sep = "\t")
            }, questions = [
                T1A(1, "Task 1.A (clean_data)"),
                T3A(1, "Task 3.A (create_classifiers)"),
                T3B(1, "Task 3.B (cross_fold_validation)"),
                T3C(1, "Task 3.C (significance_test)"),
                autograder.style.Style(kwargs.get('input_dir'), max_points = 1),
            ], **kwargs)

class T1A(autograder.question.Question):
    def score_question(self, submission, data):
        result = submission.__all__.clean_data(data)
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, pandas.DataFrame)):
            self.fail("Answer must be a DataFrame.")
            return

        self.full_credit()

class T3A(autograder.question.Question):
    def score_question(self, submission, data):
        result = submission.__all__.create_classifiers()
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, list)):
            self.fail("Answer must be a list.")
            return

        if (len(result) != 3):
            self.fail("Answer must be a list with three elements.")
            return

        self.full_credit()

class T3B(autograder.question.Question):
    def score_question(self, submission, data):
        folds = 5
        result = submission.__all__.cross_fold_validation(sklearn.dummy.DummyClassifier(),
                data, folds)

        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, list)):
            self.fail("Answer must be a list.")
            return

        if (len(result) != folds):
            self.fail("Answer must be a list with the same number of elements as folds.")
            return

        self.full_credit()

class T3C(autograder.question.Question):
    def score_question(self, submission, data):
        result = submission.__all__.significance_test([1, 2, 3], [1, 2, 3], 0.1)
        if (self.check_not_implemented(result)):
            return

        if (not isinstance(result, (bool, numpy.bool_))):
            self.fail("Answer must be a boolean.")
            return

        self.full_credit()

def main(path):
    assignment = HO5(input_dir = path)
    result = assignment.grade()

    print("***")
    print("This is NOT an actual grade, submit to the autograder for an actual grade.")
    print("***\n")

    print(result.report())

def _load_args(args):
    exe = args.pop(0)
    if (len(args) != 1 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 %s <submission path (.py or .ipynb)>" % (exe), file = sys.stderr)
        sys.exit(1)

    path = os.path.abspath(args.pop(0))

    return path

if (__name__ == '__main__'):
    main(_load_args(list(sys.argv)))
