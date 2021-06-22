#!/usr/bin/env python

import numpy as np

class Variable():
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.in_var, f.out_var
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function():
    def __call__(self, in_var):
        x = in_var.data
        y = self.forward(x)
        self.in_var = in_var
        out_var = Variable(y)
        out_var.set_creator(self)
        self.out_var = out_var

        return out_var

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        """
        gx: gradient of x (input side), i.e. df/dx
        gy: gradient of y (output side),i.e. df/dy

        df/dx = (dy/dx) * (df/dy)
        => gx = f'(x) * gy
        """
        raise NotImplementedError()

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):
        x = self.in_var.data
        gx = np.exp(x)*gy
        return gx

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.in_var.data
        gx = 2*x*gy
        return gx

def numerical_diff(func, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = func(x0)
    y1 = func(x1)
    return (y1.data - y0.data) / (2*eps)


if __name__ == '__main__':

    def f(x):
        A = Square()
        B = Exp()
        C = Square()
        return C(B(A(x)))

    x = Variable(np.array(0.5))
    y = f(x)

    print(y.data)

    dy = numerical_diff(f, x)

    print('dy by forward:  ', dy)

    y.grad = np.array(1.0)
    y.backward()

    print('dy by backward: ', x.grad)

