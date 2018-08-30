from  core.previous_application import PreviousApplication
import sys
import pandas as pd
import matplotlib.pyplot as plt
from utility.utility import checkNaNColumns
from core.installment_payment import InstallmentPayments
from core.credit_card_balance import CreditCardPayments
from core.pos_cash_balance import POSCashBalacnce
from core.application import CreditApplication
import utility.utility as util

curr_sk_ids = ['212425', '192948', '254570', '338472', '281813', '345967']

if __name__ == "__main__":
#     ca = CreditApplication('data')
#     util.store('models/app_model.pkl', ca)
    ca = util.load('models/app_model.pkl')
    score_map = ca.scoreTest(curr_sk_ids)
    for k, v in score_map.items():
        print(k, v)
