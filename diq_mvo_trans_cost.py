"""
Name    : diq_mvo_trans_cost.py
Author  : William Smyth & Layla Abu Khalaf (adapted from https://github.com/yinsenm/gerber)
Contact : drwss.academy@gmail.com
Date    : 26/12/2025
Desc    : calculates portfolio value after rebalancing,
          accounting for transaction costs proportional
          to the dollar value traded for each asset.
"""
def bisection_method(f, a, b, tol=1e-5, max_iter=100):
    """
    bisection method for finding the root of a function.
    parameters:
    f: function   - function to be rooted.
    a: float      - lower bound of interval.
    b: float      - upper bound of interval.
    tol: float    - tolerance for convergence.
    max_iter: int - maximum number of iterations.
    returns:
    float - approximate root of the function.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("The function must have different signs at a and b.")

    for i in range(max_iter):
        c = (a + b) / 2  # midpoint
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c  # Root found

        if f(c) * f(a) < 0:
            b = c  # root is in left subinterval
        else:
            a = c  # root is in right subinterval

    raise ValueError("Maximum number of iterations reached without convergence.")

class TransCost:
    def __init__(self, c: float):
        self.c = c / 10000. # transaction cost in bps
        self.tickers = None

    def get_init_cost(self, new_weights: list, old_weights: list) -> float:
        """
        :param new_weights: A dict of asset weights representing the asset allocation after rebalancing
        :param old_weights: A dict of asset weights representing the asset allocation before rebalancing
        :return: V_{t+1} / V_{t+}
        """
        # calculate e_i for each asset, the sign of the weight change
        e = [(-1 if n_w > o_w else +1) for o_w, n_w in zip(old_weights, new_weights)]

        return (
            (1 - self.c * sum(w * ei for w, ei in zip(old_weights, e))) /
            (1 - self.c * sum(w * ei for w, ei in zip(new_weights, e)))
        )

    def cost_func(self, new_weights: list, old_weights: list, cost: float):
        e = [(-1 if n_w > o_w else +1) for o_w, n_w in zip(old_weights, new_weights)]
        return 1 - self.c * sum((old_weights[i] - new_weights[i] * cost) * e[i] for i in range(len(new_weights))) - cost

    def get_cost(self, new_weights: dict, old_weights: dict):
        """
        :param new_weights: dict of asset weights representing asset allocation after rebalancing
        :param old_weights: dict of asset weights representing asset allocation before rebalancing
        :return: 1 - V_{t+1} / V_{t+}
        """
        self.tickers   = sorted(set(new_weights.keys()).union(old_weights.keys()))
        np_new_weights = [new_weights.get(ticker, 0) for ticker in self.tickers]
        np_old_weights = [old_weights.get(ticker, 0) for ticker in self.tickers]

        if sum(np_new_weights) != 0:
            np_new_weights = [wgt / sum(np_new_weights) for wgt in np_new_weights]
        if sum(np_old_weights) != 0:
            np_old_weights = [wgt / sum(np_old_weights) for wgt in np_old_weights]

        init_cost = self.get_init_cost(np_new_weights, np_old_weights)
        check = self.cost_func(np_new_weights, np_old_weights, cost=init_cost)
        if abs(check) < 1e-10:
            return 1 - init_cost  # if approximation is good enough
        else: # else return bisection result
            return 1 - bisection_method(lambda c: self.cost_func(np_new_weights, np_old_weights, c), 0, 1)
