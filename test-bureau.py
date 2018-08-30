from  core.previous_application import PreviousApplication
import sys
import pandas as pd
import matplotlib.pyplot as plt
from utility.utility import checkNaNColumns
from core.installment_payment import InstallmentPayments
from core.credit_card_balance import CreditCardPayments
from core.pos_cash_balance import POSCashBalacnce
from core.bureau import BureauBalance
import utility.utility as util

curr_sk_ids = [106358, 113499, 112503, 2577530, 112361, 391553]

if __name__ == "__main__":
    bb = BureauBalance('data')
    util.store('models/bureau_model.pkl', bb)
    bb = util.load('models/bureau_model.pkl')
    score_map = bb.score(curr_sk_ids)
    for k, v in score_map.items():
        print(k, v)
