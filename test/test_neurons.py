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
    assert activations_derivative().sigmoid__derivative(10000000) == 0

def test_activations_derivative_m10000000():
    assert activations_derivative().sigmoid__derivative(-10000000) == 0

def test_activations_relu_positive():
    assert [ele == activations().relu(ele) for ele in range(1000)]

def test_activations_relu_negative():
    assert [0 == activations().relu(-ele) for ele in range(1000)]

def test_derivative_relu_positive():
    assert [1 == activations_derivative().relu_derivative(ele) for ele in range(1000)]

def test_derivative_relu_negative():
    assert [0 == activations_derivative().relu_derivative(-ele) for ele in range(1000)]

def test_activation_tanh_0():
    assert activations().tanh(0) == 0

def test_activation_tanh_10000000():
    assert activations().tanh(10000000) == 1

def test_activation_tanh_m10000000():
    assert activations().tanh(-10000000) == -1

def test_activations_derivative_tanh_0():
    assert activations_derivative().tanh_derivative(0) == 1

def test_activations_derivative_tanh_10000000():
    assert activations_derivative().tanh_derivative(10000000) == 0

def test_activations_derivative_tanh_m10000000():
    assert activations_derivative().tanh_derivative(-10000000) == 0

