import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import string
import random
from math import exp,sin,cos

#rozwiazanie
def compare_plot(x1:np.ndarray,y1:np.ndarray,x2:np.ndarray,y2:np.ndarray,
                 xlabel: str,ylabel:str,title:str,label1:str,label2:str):

    if x1.shape != y1.shape or min(x1.shape) == 0:
        return None
    if x2.shape != y2.shape or min(x2.shape) == 0:
        return None
    fig, ax = plt.subplots()
    ax.plot(x1,y1,'b',label=label1,linewidth=4.0)
    ax.plot(x2,y2,'r',label=label2,linewidth=2.0)
    ax.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    return fig

def parallel_plot(x1:np.ndarray,y1:np.ndarray,x2:np.ndarray,y2:np.ndarray,
                  x1label:str,y1label:str,x2label:str,y2label:str,title:str,orientation:str='-'):

    if x1.shape != y1.shape or min(x1.shape) == 0:
        return None
    if x2.shape != y2.shape or min(x2.shape) == 0:
        return None
    if x1.shape != x2.shape:
        return None
    fig = plt.figure()
    if orientation == '|':
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(221)
    elif orientation == '-':
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
    else:
        return None

    ax.plot(x1, y1)
    ax.set(xlabel=x1label, ylabel=y1label)
    ax2.plot(x2, y2)
    ax2.set(xlabel=x2label, ylabel=y2label)
    fig.suptitle("{title}".format(title=title))
    ax.grid(color='b', linestyle='-', linewidth=0.1)
    ax2.grid(color='b', linestyle='-', linewidth=0.1)
    plt.show()
    return fig


def log_plot(x:np.ndarray,y:np.ndarray,xlabel:str,ylabel:str,title:str,log_axis:str):
    if x.shape != y.shape or min(x.shape) == 0:
        return None

    if log_axis == "x":
        fig = plt.figure()
        log = fig.add_subplot(111)
        log.semilogx(x,y,label='logx')
        log.grid(color='b', linestyle='-', linewidth=0.1)
        log.set_xlabel('{xlabel}'.format(xlabel=xlabel))
        log.set_ylabel('{ylabel}'.format(ylabel=ylabel))
        fig.suptitle("{title}".format(title=title))
        fig.legend()
        plt.show()
    elif log_axis == "y":
        fig = plt.figure()
        log = fig.add_subplot(111)
        log.semilogy(x, y, label='logy')
        log.grid(color='b', linestyle='-', linewidth=0.1)
        log.set_xlabel('{xlabel}'.format(xlabel=xlabel))
        log.set_ylabel('{ylabel}'.format(ylabel=ylabel))
        fig.suptitle("{title}".format(title=title))
        fig.legend()
        plt.show()
    elif log_axis == "xy":
        fig = plt.figure()
        log = fig.add_subplot(111)
        log.loglog(x, y, label='logxy')
        log.grid(color='b', linestyle='-', linewidth=0.1)
        log.set_xlabel('{xlabel}'.format(xlabel=xlabel))
        log.set_ylabel('{ylabel}'.format(ylabel=ylabel))
        fig.suptitle("{title}".format(title=title))
        fig.legend()
        plt.show()
    else:
        return None
    return fig

