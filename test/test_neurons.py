import pytest
import numpy as np
from neural_network.neurons import activations, activations_derivative


def test_activations_sigmoid_0():
    assert activations().sigmoid(0) == 0.5

def test_activations_sigmoid_10000000():
    assert activations().sigmoid(10000000) == 1

def test_activations_sigmoid_m10000000():
    assert activations().sigmoid(-10000000) == 0

def test_activations_derivative_0():
    assert activations_derivative().sigmoid__derivative(0) == 0.25

def test_activations_derivative_10000000():
    assert activations_derivative().sigmoid__derivative(10000) == 0

def test_activations_derivative_m10000000():
    assert activations_derivative().sigmoid__derivative(-10000000) == 0

def test_aactivations_relu_positive():
    assert [ele == activations().relu(ele) for ele in range(1000)]

def test_aactivations_relu_negative():
    assert [0 == activations().relu(-ele) for ele in range(1000)]