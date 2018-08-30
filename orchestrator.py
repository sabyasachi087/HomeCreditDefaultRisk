from  core.previous_application import PreviousApplication
import sys
import pandas as pd
import matplotlib.pyplot as plt
from utility.utility import checkNaNColumns
from core.installment_payment import InstallmentPayments
from core.credit_card_balance import CreditCardPayments
from core.pos_cash_balance import POSCashBalacnce
from core.application import CreditApplication
from core.final_ensemble import FinalEnsembleModel
import utility.utility as util


if __name__ == "__main__":
#     fem = FinalEnsembleModel('data')
#     util.store('models/fem_model.pkl', fem)
    fem = util.load('models/fem_model.pkl')
    fem.setScoreType('actual')
    fem.score()   