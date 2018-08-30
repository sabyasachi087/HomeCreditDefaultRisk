from  core.previous_application import PreviousApplication
import sys
import pandas as pd
import matplotlib.pyplot as plt
from utility.utility import checkNaNColumns
from core.installment_payment import InstallmentPayments
from core.credit_card_balance import CreditCardPayments
from core.pos_cash_balance import POSCashBalacnce

ens_data = pd.read_csv('ensemble_data.csv')
columns_numeric = ['DOC', 'REALTY', 'APP', 'BUREAU', 'CCB', 'PCB', 'PREV_APP', 'INS_PMT']
checkNaNColumns(columns_numeric, ens_data)
