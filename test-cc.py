from  core.previous_application import PreviousApplication
import sys
import pandas as pd
import matplotlib.pyplot as plt
from utility.utility import checkNaNColumns
from core.installment_payment import InstallmentPayments
from core.credit_card_balance import CreditCardPayments
import utility.utility as util

curr_sk_ids = ['106358', '113499', '112503', '2577530', '112361', '391553']

if __name__ == "__main__":
#     ccb = CreditCardPayments('data')
#     util.store('models/ccb_model.pkl', ccb)
    ccb = util.load('models/ccb_model.pkl')
    score_map = ccb.score(curr_sk_ids)
    for k, v in score_map.items():
        print(k, v)
